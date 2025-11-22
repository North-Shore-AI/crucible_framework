defmodule Crucible.Tinkex.CheckpointManagerTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.CheckpointManager

  setup do
    # Create temp directory for tests
    tmp_dir = Path.join(System.tmp_dir!(), "checkpoint_manager_test_#{:rand.uniform(100_000)}")
    File.mkdir_p!(tmp_dir)

    on_exit(fn -> File.rm_rf!(tmp_dir) end)

    {:ok, manager} =
      CheckpointManager.start_link(
        experiment_id: "test-exp-123",
        storage_dir: tmp_dir,
        max_checkpoints: 5
      )

    {:ok, manager: manager, tmp_dir: tmp_dir}
  end

  describe "save/4" do
    test "saves checkpoint with metrics", %{manager: manager} do
      metrics = %{loss: 0.5, accuracy: 0.85}
      {:ok, checkpoint} = CheckpointManager.save(manager, 100, metrics)

      assert checkpoint.step == 100
      assert checkpoint.metrics == metrics
      assert checkpoint.experiment_id == "test-exp-123"
      assert String.contains?(checkpoint.name, "step_100")
    end

    test "generates unique name with timestamp", %{manager: manager} do
      {:ok, cp1} = CheckpointManager.save(manager, 100, %{loss: 0.5})
      {:ok, cp2} = CheckpointManager.save(manager, 100, %{loss: 0.4})

      # Names should be unique even for same step (includes microsecond timestamp or counter)
      assert cp1.name != cp2.name

      # Verify both checkpoints are stored and retrievable
      checkpoints = CheckpointManager.list(manager)
      checkpoint_names = Enum.map(checkpoints, & &1.name)
      assert cp1.name in checkpoint_names
      assert cp2.name in checkpoint_names
    end

    test "enforces max_checkpoints limit", %{manager: manager} do
      # Save 6 checkpoints (max is 5)
      for step <- 1..6 do
        CheckpointManager.save(manager, step * 100, %{loss: 1.0 / step})
      end

      checkpoints = CheckpointManager.list(manager)
      assert length(checkpoints) == 5
    end

    test "returns error for invalid step", %{manager: manager} do
      result = CheckpointManager.save(manager, -1, %{})
      assert {:error, :invalid_step} = result
    end
  end

  describe "list/1" do
    test "returns empty list initially", %{manager: manager} do
      assert CheckpointManager.list(manager) == []
    end

    test "returns checkpoints sorted by step descending", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5})
      CheckpointManager.save(manager, 300, %{loss: 0.3})
      CheckpointManager.save(manager, 200, %{loss: 0.4})

      checkpoints = CheckpointManager.list(manager)
      steps = Enum.map(checkpoints, & &1.step)
      assert steps == [300, 200, 100]
    end
  end

  describe "get/2" do
    test "returns checkpoint by name", %{manager: manager} do
      {:ok, checkpoint} = CheckpointManager.save(manager, 100, %{loss: 0.5})

      result = CheckpointManager.get(manager, checkpoint.name)
      assert {:ok, ^checkpoint} = result
    end

    test "returns error for unknown checkpoint", %{manager: manager} do
      result = CheckpointManager.get(manager, "nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "get_best/3" do
    test "returns checkpoint with lowest loss", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5, accuracy: 0.8})
      CheckpointManager.save(manager, 200, %{loss: 0.3, accuracy: 0.85})
      CheckpointManager.save(manager, 300, %{loss: 0.4, accuracy: 0.82})

      {:ok, best} = CheckpointManager.get_best(manager, :loss, :min)
      assert best.step == 200
      assert best.metrics.loss == 0.3
    end

    test "returns checkpoint with highest accuracy when direction is :max", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5, accuracy: 0.8})
      CheckpointManager.save(manager, 200, %{loss: 0.3, accuracy: 0.85})
      CheckpointManager.save(manager, 300, %{loss: 0.4, accuracy: 0.9})

      {:ok, best} = CheckpointManager.get_best(manager, :accuracy, :max)
      assert best.step == 300
      assert best.metrics.accuracy == 0.9
    end

    test "returns error when no checkpoints exist", %{manager: manager} do
      result = CheckpointManager.get_best(manager, :loss, :min)
      assert {:error, :no_checkpoints} = result
    end

    test "returns error for missing metric", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5})

      result = CheckpointManager.get_best(manager, :missing_metric, :min)
      assert {:error, :metric_not_found} = result
    end
  end

  describe "prune/3" do
    test "keeps only top N checkpoints by metric", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5})
      CheckpointManager.save(manager, 200, %{loss: 0.3})
      CheckpointManager.save(manager, 300, %{loss: 0.4})
      CheckpointManager.save(manager, 400, %{loss: 0.2})
      CheckpointManager.save(manager, 500, %{loss: 0.6})

      {:ok, pruned_count} = CheckpointManager.prune(manager, 2, :loss)

      assert pruned_count == 3

      checkpoints = CheckpointManager.list(manager)
      assert length(checkpoints) == 2

      losses = Enum.map(checkpoints, & &1.metrics.loss)
      assert 0.2 in losses
      assert 0.3 in losses
    end

    test "returns 0 when nothing to prune", %{manager: manager} do
      CheckpointManager.save(manager, 100, %{loss: 0.5})

      {:ok, pruned_count} = CheckpointManager.prune(manager, 5, :loss)
      assert pruned_count == 0
    end
  end

  describe "delete/2" do
    test "deletes checkpoint by name", %{manager: manager} do
      {:ok, checkpoint} = CheckpointManager.save(manager, 100, %{loss: 0.5})

      assert :ok = CheckpointManager.delete(manager, checkpoint.name)
      assert {:error, :not_found} = CheckpointManager.get(manager, checkpoint.name)
    end

    test "returns error for unknown checkpoint", %{manager: manager} do
      result = CheckpointManager.delete(manager, "nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "load_for_sampling/3" do
    test "returns checkpoint data for sampling", %{manager: manager} do
      {:ok, checkpoint} = CheckpointManager.save(manager, 100, %{loss: 0.5})

      # This would normally download from Tinkex, but in tests we just verify the path
      result = CheckpointManager.load_for_sampling(manager, checkpoint.name, [])

      case result do
        {:ok, _} -> assert true
        # Expected in test env
        {:error, :not_downloaded} -> assert true
      end
    end
  end
end
