defmodule Crucible.Ensemble.ModelEnsembleTest do
  use ExUnit.Case, async: true

  alias Crucible.Ensemble.ModelEnsemble
  alias Crucible.Ensemble.AdapterPool

  describe "create/2" do
    test "creates ensemble with default options" do
      {:ok, pool} = AdapterPool.start_link([])

      {:ok, ensemble} =
        ModelEnsemble.create("test-ensemble",
          pool: pool,
          strategy: :weighted,
          execution_mode: :parallel
        )

      assert ensemble.name == "test-ensemble"
      assert ensemble.strategy == :weighted
      assert ensemble.execution_mode == :parallel
    end

    test "creates ensemble with custom timeout and hedging config" do
      {:ok, pool} = AdapterPool.start_link([])

      {:ok, ensemble} =
        ModelEnsemble.create("custom-ensemble",
          pool: pool,
          timeout: 60_000,
          hedging_config: %{strategy: :percentile_75, delay_ms: 100}
        )

      assert ensemble.timeout == 60_000
      assert ensemble.hedging_config.strategy == :percentile_75
    end
  end

  # from_registry/3 tests removed - ModelRegistry was deleted as part of Tinkex abstraction cleanup

  describe "add_model/2" do
    test "adds a model to existing ensemble" do
      {:ok, pool} = AdapterPool.start_link([])
      {:ok, ensemble} = ModelEnsemble.create("test", pool: pool)

      model_spec = %{
        name: "new-model",
        checkpoint_path: "path/new",
        weight: 0.5,
        tags: [:added]
      }

      {:ok, updated} = ModelEnsemble.add_model(ensemble, model_spec)

      clients = AdapterPool.all_clients(updated.pool)
      names = Enum.map(clients, fn {adapter, _} -> adapter.name end)
      assert "new-model" in names
    end
  end

  describe "remove_model/2" do
    test "removes a model from ensemble" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter = %{name: "model-1", checkpoint_path: "path/1", weight: 0.5, tags: [:test]}
      :ok = AdapterPool.add_client(pool, adapter, make_ref())

      {:ok, ensemble} = ModelEnsemble.create("test", pool: pool)
      {:ok, updated} = ModelEnsemble.remove_model(ensemble, "model-1")

      clients = AdapterPool.all_clients(updated.pool)
      assert clients == []
    end
  end

  describe "get_stats/1" do
    test "returns ensemble statistics" do
      {:ok, pool} = AdapterPool.start_link([])

      adapter1 = %{name: "m1", checkpoint_path: "p1", weight: 0.6, tags: [:prod]}
      adapter2 = %{name: "m2", checkpoint_path: "p2", weight: 0.4, tags: [:prod]}
      :ok = AdapterPool.add_client(pool, adapter1, make_ref())
      :ok = AdapterPool.add_client(pool, adapter2, make_ref())

      {:ok, ensemble} = ModelEnsemble.create("test", pool: pool)

      stats = ModelEnsemble.get_stats(ensemble)

      assert stats.model_count == 2
      assert stats.total_weight == 1.0
      assert "m1" in stats.model_names
      assert "m2" in stats.model_names
    end
  end
end
