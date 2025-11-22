defmodule Crucible.Tinkex.ModelRegistryTest do
  use ExUnit.Case, async: true

  alias Crucible.Tinkex.ModelRegistry

  setup do
    {:ok, registry} = ModelRegistry.start_link()
    {:ok, registry: registry}
  end

  describe "register/3" do
    test "registers model with metadata", %{registry: registry} do
      metadata = %{
        experiment_id: "exp-123",
        checkpoint_name: "step_1000",
        checkpoint_path: "tinker://exp-123/checkpoints/step_1000",
        base_model: "llama-3-8b",
        lora_rank: 16,
        metrics: %{loss: 0.25, accuracy: 0.92},
        tags: [:production, :scifact]
      }

      assert :ok = ModelRegistry.register(registry, "scifact-v1", metadata)

      {:ok, model} = ModelRegistry.get(registry, "scifact-v1")
      assert model.name == "scifact-v1"
      assert model.experiment_id == "exp-123"
      assert model.metrics.accuracy == 0.92
      assert :production in model.tags
    end

    test "prevents duplicate names", %{registry: registry} do
      metadata = %{experiment_id: "exp-1", checkpoint_path: "path1", metrics: %{}}

      assert :ok = ModelRegistry.register(registry, "model-1", metadata)
      assert {:error, :already_exists} = ModelRegistry.register(registry, "model-1", metadata)
    end

    test "validates required fields", %{registry: registry} do
      # Missing experiment_id
      result = ModelRegistry.register(registry, "model-1", %{})
      assert {:error, :missing_required_fields} = result
    end
  end

  describe "get/2" do
    test "returns model by name", %{registry: registry} do
      metadata = %{experiment_id: "exp-1", checkpoint_path: "path1", metrics: %{loss: 0.3}}
      ModelRegistry.register(registry, "model-1", metadata)

      {:ok, model} = ModelRegistry.get(registry, "model-1")
      assert model.name == "model-1"
    end

    test "returns error for unknown model", %{registry: registry} do
      result = ModelRegistry.get(registry, "nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "list/2" do
    test "returns all models", %{registry: registry} do
      for i <- 1..3 do
        ModelRegistry.register(registry, "model-#{i}", %{
          experiment_id: "exp-#{i}",
          checkpoint_path: "path-#{i}",
          metrics: %{loss: i * 0.1}
        })
      end

      models = ModelRegistry.list(registry)
      assert length(models) == 3
    end

    test "filters by experiment_id", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-a",
        checkpoint_path: "p1",
        metrics: %{}
      })

      ModelRegistry.register(registry, "model-2", %{
        experiment_id: "exp-b",
        checkpoint_path: "p2",
        metrics: %{}
      })

      ModelRegistry.register(registry, "model-3", %{
        experiment_id: "exp-a",
        checkpoint_path: "p3",
        metrics: %{}
      })

      models = ModelRegistry.list(registry, experiment_id: "exp-a")
      assert length(models) == 2
    end
  end

  describe "find_by_tags/2" do
    test "finds models with matching tags", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{},
        tags: [:production, :scifact]
      })

      ModelRegistry.register(registry, "model-2", %{
        experiment_id: "exp-2",
        checkpoint_path: "p2",
        metrics: %{},
        tags: [:staging, :scifact]
      })

      ModelRegistry.register(registry, "model-3", %{
        experiment_id: "exp-3",
        checkpoint_path: "p3",
        metrics: %{},
        tags: [:production, :other]
      })

      models = ModelRegistry.find_by_tags(registry, [:production])
      assert length(models) == 2

      names = Enum.map(models, & &1.name)
      assert "model-1" in names
      assert "model-3" in names
    end

    test "requires all tags to match", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{},
        tags: [:production, :scifact]
      })

      ModelRegistry.register(registry, "model-2", %{
        experiment_id: "exp-2",
        checkpoint_path: "p2",
        metrics: %{},
        tags: [:production]
      })

      models = ModelRegistry.find_by_tags(registry, [:production, :scifact])
      assert length(models) == 1
      assert hd(models).name == "model-1"
    end
  end

  describe "find_for_ensemble/2" do
    test "finds top N models by metric", %{registry: registry} do
      for i <- 1..5 do
        ModelRegistry.register(registry, "model-#{i}", %{
          experiment_id: "exp-#{i}",
          checkpoint_path: "path-#{i}",
          metrics: %{accuracy: 0.7 + i * 0.05},
          tags: [:ensemble_candidate]
        })
      end

      {:ok, models} =
        ModelRegistry.find_for_ensemble(registry, %{
          sort_by: :accuracy,
          top_n: 3
        })

      assert length(models) == 3
      accuracies = Enum.map(models, & &1.metrics.accuracy)
      # Use approximate comparison for floating point
      assert_in_delta hd(accuracies), 0.95, 0.001
      assert_in_delta Enum.at(accuracies, 1), 0.90, 0.001
      assert_in_delta Enum.at(accuracies, 2), 0.85, 0.001
    end

    test "filters by tags", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{accuracy: 0.9},
        tags: [:candidate]
      })

      ModelRegistry.register(registry, "model-2", %{
        experiment_id: "exp-2",
        checkpoint_path: "p2",
        metrics: %{accuracy: 0.95},
        tags: []
      })

      {:ok, models} =
        ModelRegistry.find_for_ensemble(registry, %{
          sort_by: :accuracy,
          top_n: 3,
          tags: [:candidate]
        })

      assert length(models) == 1
      assert hd(models).name == "model-1"
    end

    test "returns models sorted by performance", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{accuracy: 0.8}
      })

      ModelRegistry.register(registry, "model-2", %{
        experiment_id: "exp-2",
        checkpoint_path: "p2",
        metrics: %{accuracy: 0.9}
      })

      {:ok, models} =
        ModelRegistry.find_for_ensemble(registry, %{
          sort_by: :accuracy,
          top_n: 2
        })

      assert hd(models).metrics.accuracy == 0.9
    end
  end

  describe "update_metrics/3" do
    test "updates model metrics", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{loss: 0.5}
      })

      :ok = ModelRegistry.update_metrics(registry, "model-1", %{loss: 0.3, accuracy: 0.9})

      {:ok, model} = ModelRegistry.get(registry, "model-1")
      assert model.metrics.loss == 0.3
      assert model.metrics.accuracy == 0.9
    end

    test "returns error for unknown model", %{registry: registry} do
      result = ModelRegistry.update_metrics(registry, "nonexistent", %{})
      assert {:error, :not_found} = result
    end
  end

  describe "delete/2" do
    test "deletes model from registry", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{}
      })

      assert :ok = ModelRegistry.delete(registry, "model-1")
      assert {:error, :not_found} = ModelRegistry.get(registry, "model-1")
    end

    test "returns error for unknown model", %{registry: registry} do
      result = ModelRegistry.delete(registry, "nonexistent")
      assert {:error, :not_found} = result
    end
  end

  describe "export/2 and import/2" do
    test "exports registry to JSON", %{registry: registry} do
      ModelRegistry.register(registry, "model-1", %{
        experiment_id: "exp-1",
        checkpoint_path: "p1",
        metrics: %{loss: 0.3},
        tags: [:production]
      })

      tmp_path = Path.join(System.tmp_dir!(), "registry_export_#{:rand.uniform(100_000)}.json")

      try do
        :ok = ModelRegistry.export(registry, tmp_path)
        assert File.exists?(tmp_path)

        content = File.read!(tmp_path)
        data = Jason.decode!(content)
        assert length(data["models"]) == 1
      after
        File.rm(tmp_path)
      end
    end

    test "imports registry from JSON", %{registry: registry} do
      data = %{
        "version" => 1,
        "models" => [
          %{
            "name" => "imported-model",
            "experiment_id" => "exp-1",
            "checkpoint_path" => "p1",
            "metrics" => %{"loss" => 0.3},
            "tags" => ["imported"],
            "created_at" => DateTime.to_iso8601(DateTime.utc_now())
          }
        ]
      }

      tmp_path = Path.join(System.tmp_dir!(), "registry_import_#{:rand.uniform(100_000)}.json")
      File.write!(tmp_path, Jason.encode!(data))

      try do
        :ok = ModelRegistry.import(registry, tmp_path)

        {:ok, model} = ModelRegistry.get(registry, "imported-model")
        assert model.experiment_id == "exp-1"
      after
        File.rm(tmp_path)
      end
    end
  end
end
