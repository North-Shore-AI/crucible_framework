defmodule Crucible.Tinkex.ArtifactStoreTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.ArtifactStore

  setup do
    # Create temp cache directory for tests
    tmp_dir = Path.join(System.tmp_dir!(), "artifact_store_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(tmp_dir)

    # Override cache dir for tests
    Application.put_env(:crucible, :artifact_cache_dir, tmp_dir)

    on_exit(fn ->
      Application.delete_env(:crucible, :artifact_cache_dir)
      File.rm_rf!(tmp_dir)
    end)

    {:ok, tmp_dir: tmp_dir}
  end

  describe "cache_dir/0" do
    test "returns configured cache directory", %{tmp_dir: tmp_dir} do
      assert ArtifactStore.cache_dir() == tmp_dir
    end
  end

  describe "store/3" do
    test "stores artifact with checksum", %{tmp_dir: _tmp_dir} do
      content = "test artifact content"

      {:ok, artifact} = ArtifactStore.store("test-artifact", content)

      assert artifact.name == "test-artifact"
      assert artifact.size_bytes == byte_size(content)
      assert artifact.checksum != nil
      assert File.exists?(artifact.local_path)
    end

    test "creates cache directory if missing" do
      new_dir = Path.join(System.tmp_dir!(), "new_cache_#{:rand.uniform(100_000)}")
      Application.put_env(:crucible, :artifact_cache_dir, new_dir)

      try do
        content = "test content"
        {:ok, artifact} = ArtifactStore.store("test-artifact", content)

        assert File.exists?(artifact.local_path)
      after
        File.rm_rf!(new_dir)
      end
    end

    test "calculates correct checksum" do
      content = "deterministic content"

      {:ok, artifact1} = ArtifactStore.store("artifact-1", content)
      {:ok, artifact2} = ArtifactStore.store("artifact-2", content)

      assert artifact1.checksum == artifact2.checksum
    end

    test "stores binary content" do
      binary_content = <<0, 1, 2, 3, 255, 254, 253>>

      {:ok, artifact} = ArtifactStore.store("binary-artifact", binary_content)

      stored_content = File.read!(artifact.local_path)
      assert stored_content == binary_content
    end
  end

  describe "get/1" do
    test "returns stored artifact" do
      {:ok, stored} = ArtifactStore.store("my-artifact", "content")

      {:ok, artifact} = ArtifactStore.get("my-artifact")
      assert artifact.name == "my-artifact"
      assert artifact.local_path == stored.local_path
    end

    test "returns error for unknown artifact" do
      result = ArtifactStore.get("nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "exists?/1" do
    test "returns true for existing artifact" do
      ArtifactStore.store("existing", "content")
      assert ArtifactStore.exists?("existing") == true
    end

    test "returns false for unknown artifact" do
      assert ArtifactStore.exists?("unknown") == false
    end
  end

  describe "delete/1" do
    test "deletes artifact and file" do
      {:ok, artifact} = ArtifactStore.store("to-delete", "content")
      local_path = artifact.local_path

      assert :ok = ArtifactStore.delete("to-delete")
      assert ArtifactStore.exists?("to-delete") == false
      assert File.exists?(local_path) == false
    end

    test "returns error for unknown artifact" do
      result = ArtifactStore.delete("nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "list/0" do
    test "returns all stored artifacts" do
      ArtifactStore.store("artifact-1", "content 1")
      ArtifactStore.store("artifact-2", "content 2")
      ArtifactStore.store("artifact-3", "content 3")

      artifacts = ArtifactStore.list()
      assert length(artifacts) == 3

      names = Enum.map(artifacts, & &1.name)
      assert "artifact-1" in names
      assert "artifact-2" in names
      assert "artifact-3" in names
    end

    test "returns empty list when no artifacts" do
      assert ArtifactStore.list() == []
    end
  end

  describe "total_size/0" do
    test "returns total size of all artifacts" do
      ArtifactStore.store("artifact-1", String.duplicate("a", 100))
      ArtifactStore.store("artifact-2", String.duplicate("b", 200))

      total = ArtifactStore.total_size()
      assert total == 300
    end

    test "returns 0 when no artifacts" do
      assert ArtifactStore.total_size() == 0
    end
  end

  describe "cleanup/1" do
    test "removes oldest artifacts when over size limit" do
      # Store artifacts sequentially - timestamps are assigned on creation
      {:ok, _} = ArtifactStore.store("old-artifact", String.duplicate("a", 100))
      {:ok, _} = ArtifactStore.store("medium-artifact", String.duplicate("b", 100))
      {:ok, _} = ArtifactStore.store("new-artifact", String.duplicate("c", 100))

      # Cleanup to 200 bytes (should remove oldest)
      {:ok, removed_count} = ArtifactStore.cleanup(200)

      assert removed_count == 1
      assert ArtifactStore.exists?("old-artifact") == false
      assert ArtifactStore.exists?("medium-artifact") == true
      assert ArtifactStore.exists?("new-artifact") == true
    end

    test "preserves most recent artifacts" do
      {:ok, _} = ArtifactStore.store("old", String.duplicate("x", 50))
      {:ok, _} = ArtifactStore.store("new", String.duplicate("y", 50))

      {:ok, _} = ArtifactStore.cleanup(50)

      assert ArtifactStore.exists?("new") == true
    end

    test "returns 0 when already under limit" do
      ArtifactStore.store("small", "tiny")

      {:ok, removed} = ArtifactStore.cleanup(1000)
      assert removed == 0
    end
  end

  describe "verify_checksum/2" do
    test "returns ok for matching checksum" do
      {:ok, artifact} = ArtifactStore.store("verified", "content")

      result = ArtifactStore.verify_checksum(artifact.local_path, artifact.checksum)
      assert :ok = result
    end

    test "returns error for mismatched checksum" do
      {:ok, artifact} = ArtifactStore.store("verified", "content")

      result = ArtifactStore.verify_checksum(artifact.local_path, "wrong_checksum")
      assert {:error, :checksum_mismatch} = result
    end

    test "returns error for missing file" do
      result = ArtifactStore.verify_checksum("/nonexistent/path", "any")
      assert {:error, :file_not_found} = result
    end
  end

  describe "download_checkpoint/2" do
    test "returns error when tinkex client not configured" do
      # This tests the error path since we don't have a real Tinkex client
      result = ArtifactStore.download_checkpoint("tinker://test/checkpoint")

      # Expected to fail without proper Tinkex setup
      assert {:error, _reason} = result
    end
  end
end
