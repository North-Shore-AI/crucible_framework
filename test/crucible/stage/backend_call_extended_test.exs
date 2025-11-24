defmodule Crucible.Stage.BackendCallExtendedTest do
  use ExUnit.Case, async: true

  import Mox

  alias Crucible.Context
  alias Crucible.Stage.BackendCallExtended
  alias Crucible.IR.{BackendRef, Experiment, EnsembleConfig, HedgingConfig, ReliabilityConfig}

  setup :set_mox_from_context
  setup :verify_on_exit!

  setup do
    Application.put_env(:crucible_framework, :backends, %{
      mock: Crucible.BackendMock,
      mock2: Crucible.BackendMock,
      mock3: Crucible.BackendMock
    })

    :ok
  end

  describe "ensemble support" do
    test "executes ensemble voting with multiple backends" do
      # Setup multiple backend mocks
      expect(Crucible.BackendMock, :init, 3, fn _id, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :majority_vote,
            members: [
              %BackendRef{id: :mock, options: %{}},
              %BackendRef{id: :mock2, options: %{}},
              %BackendRef{id: :mock3, options: %{}}
            ],
            options: %{}
          },
          hedging: %HedgingConfig{strategy: :off}
        }
      }

      expect(Crucible.BackendMock, :start_session, 3, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :save_checkpoint, 3, fn :session, 0 ->
        {:ok, :checkpoint}
      end)

      expect(Crucible.BackendMock, :create_sampler, 3, fn :session, :checkpoint ->
        {:ok, :sampler}
      end)

      # Mock different responses from each backend
      expect(Crucible.BackendMock, :sample, 3, fn :sampler, "test prompt", _opts ->
        # Return same answer from 2 backends for majority
        {:ok, ["answer A"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test prompt"]
               })

      # Check ensemble metrics were recorded
      assert Map.has_key?(new_ctx.metrics, :ensemble)
      ensemble_metrics = new_ctx.metrics.ensemble
      assert ensemble_metrics.strategy == :majority_vote
      assert ensemble_metrics.members_count == 3
      assert ensemble_metrics.samples == 1

      # Check output was recorded
      assert length(new_ctx.outputs) == 1
    end

    test "handles ensemble member failures gracefully" do
      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :majority_vote,
            members: [
              %BackendRef{id: :mock, options: %{}},
              %BackendRef{id: :mock2, options: %{}}
            ],
            options: %{}
          }
        }
      }

      # First backend succeeds
      expect(Crucible.BackendMock, :init, 1, fn :mock, %{} -> {:ok, :state} end)

      expect(Crucible.BackendMock, :start_session, 1, fn :state, ^experiment ->
        {:ok, :session}
      end)

      # Second backend fails
      expect(Crucible.BackendMock, :init, 1, fn :mock2, %{} -> {:error, :init_failed} end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        outputs: []
      }

      assert {:error, {:ensemble_init_failed, _}} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })
    end

    test "applies weighted voting strategy" do
      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :weighted,
            members: [
              %BackendRef{id: :mock, options: %{}}
            ],
            options: %{weights: %{mock: 1.0}}
          }
        }
      }

      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :save_checkpoint, fn :session, 0 ->
        {:ok, :checkpoint}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, :checkpoint ->
        {:ok, :sampler}
      end)

      expect(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["weighted answer"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      assert new_ctx.metrics.ensemble.strategy == :weighted
    end

    test "ensemble mode requires sample mode, not training" do
      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :majority_vote,
            members: [%BackendRef{id: :mock, options: %{}}],
            options: %{}
          }
        }
      }

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        batches: [[%{input: "a", output: "b"}]]
      }

      assert {:error, :ensemble_requires_sample_mode} =
               BackendCallExtended.run(ctx, %{
                 mode: :train
               })
    end
  end

  describe "hedging support" do
    test "applies fixed delay hedging strategy" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{strategy: :none},
          hedging: %HedgingConfig{
            strategy: :fixed_delay,
            delay_ms: 50,
            max_extra_requests: 1,
            options: %{}
          }
        }
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, nil ->
        {:ok, :sampler}
      end)

      stub(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["hedged response"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        assigns: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      # Check hedging metrics were recorded
      assert Map.has_key?(new_ctx.metrics.backend, :hedging)
      hedging_metrics = new_ctx.metrics.backend.hedging
      assert hedging_metrics.enabled == true
    end

    test "applies percentile hedging strategy" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          hedging: %HedgingConfig{
            strategy: :percentile,
            percentile: 95,
            max_extra_requests: 1,
            options: %{}
          }
        }
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, nil ->
        {:ok, :sampler}
      end)

      stub(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["response"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        assigns: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      assert new_ctx.metrics.backend.hedging.enabled == true
    end

    test "hedging works with ensemble" do
      # Setup for 2 backends with hedging
      expect(Crucible.BackendMock, :init, 2, fn _id, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :majority_vote,
            members: [
              %BackendRef{id: :mock, options: %{}},
              %BackendRef{id: :mock2, options: %{}}
            ],
            options: %{}
          },
          hedging: %HedgingConfig{
            strategy: :fixed_delay,
            delay_ms: 10,
            max_extra_requests: 1,
            options: %{}
          }
        }
      }

      expect(Crucible.BackendMock, :start_session, 2, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :save_checkpoint, 2, fn :session, 0 ->
        {:ok, :checkpoint}
      end)

      expect(Crucible.BackendMock, :create_sampler, 2, fn :session, :checkpoint ->
        {:ok, :sampler}
      end)

      stub(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["ensemble+hedge answer"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      assert Map.has_key?(new_ctx.metrics, :ensemble)
      assert new_ctx.metrics.ensemble.members_count == 2
    end
  end

  describe "backward compatibility" do
    test "works without ensemble or hedging (original behavior)" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{strategy: :none},
          hedging: %HedgingConfig{strategy: :off}
        }
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :train_step, fn :session, _batch ->
        {:ok, %{loss: 0.1, batch_size: 1, metrics: %{}}}
      end)

      expect(Crucible.BackendMock, :save_checkpoint, fn :session, 1 ->
        {:ok, :checkpoint}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, :checkpoint ->
        {:ok, :sampler}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        batches: [[%{input: "a", output: "b"}]],
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :train,
                 create_sampler?: true
               })

      # Should work exactly like original BackendCall
      assert new_ctx.metrics.backend.mean_loss == 0.1
      assert new_ctx.assigns.checkpoint_ref == :checkpoint
    end

    test "sampling without ensemble or hedging" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{}
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, nil ->
        {:ok, :sampler}
      end)

      expect(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["regular response"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        assigns: %{},
        outputs: []
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      assert new_ctx.outputs == [%{prompt: "test", responses: ["regular response"]}]
      assert new_ctx.metrics.backend.samples == 1
    end
  end

  describe "trace integration" do
    test "adds trace events when tracing is enabled" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{
          ensemble: %EnsembleConfig{
            strategy: :majority_vote,
            members: [%BackendRef{id: :mock, options: %{}}],
            options: %{}
          }
        }
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :save_checkpoint, fn :session, 0 ->
        {:ok, :checkpoint}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, :checkpoint ->
        {:ok, :sampler}
      end)

      expect(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["traced answer"]}
      end)

      # Create a context with trace enabled
      trace = CrucibleTrace.new_chain("test experiment")

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        outputs: [],
        trace: trace
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      # Check that trace events were added
      assert new_ctx.trace != nil
      assert length(new_ctx.trace.events) > length(trace.events)
    end

    test "works without trace (trace is nil)" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{}
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      expect(Crucible.BackendMock, :create_sampler, fn :session, nil ->
        {:ok, :sampler}
      end)

      expect(Crucible.BackendMock, :sample, fn :sampler, "test", _opts ->
        {:ok, ["answer"]}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{},
        assigns: %{},
        outputs: [],
        # No trace
        trace: nil
      }

      assert {:ok, new_ctx} =
               BackendCallExtended.run(ctx, %{
                 mode: :sample,
                 prompts: ["test"]
               })

      # Should work fine without trace
      assert new_ctx.trace == nil
      assert new_ctx.outputs == [%{prompt: "test", responses: ["answer"]}]
    end
  end

  describe "error handling" do
    test "handles missing backend gracefully" do
      ctx = %Context{
        experiment_id: "exp",
        run_id: "run",
        experiment: %Experiment{
          id: "exp",
          # No backend
          backend: nil,
          pipeline: []
        }
      }

      assert {:error, :missing_backend} = BackendCallExtended.run(ctx, %{})
    end

    test "handles unknown mode" do
      expect(Crucible.BackendMock, :init, fn :mock, %{} -> {:ok, :state} end)

      experiment = %Experiment{
        id: "exp",
        backend: %BackendRef{id: :mock, options: %{}},
        pipeline: [],
        reliability: %ReliabilityConfig{}
      }

      expect(Crucible.BackendMock, :start_session, fn :state, ^experiment ->
        {:ok, :session}
      end)

      ctx = %Context{
        experiment_id: experiment.id,
        run_id: "run",
        experiment: experiment,
        backend_state: %{},
        backend_sessions: %{}
      }

      assert {:error, {:unknown_mode, :invalid}} =
               BackendCallExtended.run(ctx, %{
                 mode: :invalid
               })
    end
  end
end
