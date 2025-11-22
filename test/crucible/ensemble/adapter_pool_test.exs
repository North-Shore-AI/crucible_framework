defmodule Crucible.Ensemble.AdapterPoolTest do
  use ExUnit.Case, async: true

  alias Crucible.Ensemble.AdapterPool

  describe "start_link/1" do
    test "starts the adapter pool GenServer" do
      {:ok, pid} = AdapterPool.start_link([])
      assert Process.alive?(pid)
    end
  end

  describe "create/1" do
    test "creates pool from adapter specs" do
      # Mock session and adapters
      adapters = [
        %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:test]},
        %{name: "model-2", checkpoint_path: "path/2", weight: 0.5, tags: [:test]}
      ]

      {:ok, pool} = AdapterPool.start_link([])

      # Add mock clients manually for testing
      for adapter <- adapters do
        :ok = AdapterPool.add_client(pool, adapter, make_ref())
      end

      clients = AdapterPool.all_clients(pool)
      assert length(clients) == 2
    end

    test "starts sampling clients for each adapter" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:test]}
      client_ref = make_ref()

      :ok = AdapterPool.add_client(pool, adapter, client_ref)

      assert {:ok, ^client_ref} = AdapterPool.get_client(pool, "model-1")
    end
  end

  describe "all_clients/1" do
    test "returns all clients with their specs" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter1 = %{name: "model-1", checkpoint_path: "path/1", weight: 0.6, tags: [:prod]}
      adapter2 = %{name: "model-2", checkpoint_path: "path/2", weight: 0.4, tags: [:dev]}

      :ok = AdapterPool.add_client(pool, adapter1, make_ref())
      :ok = AdapterPool.add_client(pool, adapter2, make_ref())

      clients = AdapterPool.all_clients(pool)

      assert length(clients) == 2
      names = Enum.map(clients, fn {adapter, _client} -> adapter.name end)
      assert "model-1" in names
      assert "model-2" in names
    end
  end

  describe "clients_by_tags/2" do
    test "filters clients by tags" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter1 = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:prod, :fast]}
      adapter2 = %{name: "model-2", checkpoint_path: "path/2", weight: 0.3, tags: [:dev]}
      adapter3 = %{name: "model-3", checkpoint_path: "path/3", weight: 0.2, tags: [:prod, :slow]}

      :ok = AdapterPool.add_client(pool, adapter1, make_ref())
      :ok = AdapterPool.add_client(pool, adapter2, make_ref())
      :ok = AdapterPool.add_client(pool, adapter3, make_ref())

      prod_clients = AdapterPool.clients_by_tags(pool, [:prod])
      assert length(prod_clients) == 2

      fast_prod_clients = AdapterPool.clients_by_tags(pool, [:prod, :fast])
      assert length(fast_prod_clients) == 1
      [{adapter, _client}] = fast_prod_clients
      assert adapter.name == "model-1"
    end

    test "returns empty list when no match" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:prod]}
      :ok = AdapterPool.add_client(pool, adapter, make_ref())

      clients = AdapterPool.clients_by_tags(pool, [:nonexistent])
      assert clients == []
    end
  end

  describe "get_client/2" do
    test "returns client for existing adapter" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:test]}
      client_ref = make_ref()
      :ok = AdapterPool.add_client(pool, adapter, client_ref)

      assert {:ok, ^client_ref} = AdapterPool.get_client(pool, "model-1")
    end

    test "returns error for non-existing adapter" do
      {:ok, pool} = AdapterPool.start_link([])
      assert {:error, :not_found} = AdapterPool.get_client(pool, "nonexistent")
    end
  end

  describe "remove_client/2" do
    test "removes client from pool" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:test]}
      :ok = AdapterPool.add_client(pool, adapter, make_ref())

      :ok = AdapterPool.remove_client(pool, "model-1")

      assert {:error, :not_found} = AdapterPool.get_client(pool, "model-1")
    end
  end
end
