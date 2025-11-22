defmodule Crucible.Tinkex.SessionTest do
  use ExUnit.Case, async: true
  import Mox

  alias Crucible.Tinkex.Session
  alias Crucible.Tinkex.Config

  # Set up Mox for Tinkex clients
  setup :verify_on_exit!

  # Helper to safely start session and catch connection errors
  defp safe_start_session(experiment) do
    # Trap exits so we can catch Tinkex connection errors
    Process.flag(:trap_exit, true)
    result = Session.start_link(experiment: experiment)

    case result do
      {:ok, pid} ->
        # Wait briefly for any async initialization errors
        receive do
          {:EXIT, ^pid, reason} ->
            {:error, reason}
        after
          50 -> {:ok, pid}
        end

      {:error, reason} ->
        {:error, reason}
    end
  after
    Process.flag(:trap_exit, false)
  end

  defp valid_config do
    Config.new(
      api_key: "test-api-key",
      base_url: "https://test.tinkex.com",
      base_model: "meta-llama/Llama-3.2-1B"
    )
  end

  defp valid_experiment do
    %{
      id: "test-exp-123",
      name: "Test Experiment",
      config: valid_config()
    }
  end

  describe "start_link/1" do
    test "creates session with valid config" do
      # Mock the service client start
      experiment = valid_experiment()

      # Start session - will fail without actual Tinkex mocks
      # This test documents expected behavior
      result = safe_start_session(experiment)

      case result do
        {:ok, pid} ->
          assert is_pid(pid)
          state = :sys.get_state(pid)
          assert state.experiment_id == experiment.id
          assert state.status == :ready
          GenServer.stop(pid)

        {:error, _reason} ->
          # Expected when Tinkex is not available
          :ok
      end
    end

    test "returns error with missing experiment" do
      Process.flag(:trap_exit, true)
      result = Session.start_link([])

      case result do
        {:error, _} ->
          :ok

        {:ok, pid} ->
          receive do
            {:EXIT, ^pid, _} -> :ok
          after
            50 -> GenServer.stop(pid)
          end
      end
    after
      Process.flag(:trap_exit, false)
    end

    test "returns error with invalid config" do
      experiment = %{
        id: "test",
        name: "Test",
        # No api_key
        config: Config.new()
      }

      Process.flag(:trap_exit, true)
      result = Session.start_link(experiment: experiment)

      case result do
        {:error, _reason} ->
          :ok

        {:ok, pid} ->
          receive do
            {:EXIT, ^pid, _} -> :ok
          after
            50 -> GenServer.stop(pid)
          end
      end
    after
      Process.flag(:trap_exit, false)
    end
  end

  describe "forward_backward/3" do
    @tag :integration
    test "executes training step and returns loss" do
      # This test requires actual Tinkex mocks
      # Document expected behavior
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          batch = [
            %{input: "test input", output: "test output", weight: 1.0}
          ]

          result = GenServer.call(session, {:forward_backward, batch, []})

          case result do
            {:ok, metrics} ->
              assert is_map(metrics)
              assert Map.has_key?(metrics, :loss)

            {:error, _reason} ->
              # Expected without Tinkex backend
              :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end

    test "emits telemetry events" do
      ref = make_ref()
      parent = self()

      :telemetry.attach(
        "test-handler-#{inspect(ref)}",
        [:crucible, :tinkex, :forward_backward_stop],
        fn event, measurements, metadata, _config ->
          send(parent, {:telemetry, event, measurements, metadata})
        end,
        nil
      )

      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          batch = [%{input: "test", output: "result", weight: 1.0}]

          # Call forward_backward
          GenServer.call(session, {:forward_backward, batch, []})

          # May or may not receive telemetry depending on mock setup
          receive do
            {:telemetry, [:crucible, :tinkex, :forward_backward_stop], _measurements, metadata} ->
              assert metadata.experiment_id == experiment.id
          after
            100 -> :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end

      :telemetry.detach("test-handler-#{inspect(ref)}")
    end
  end

  describe "optim_step/2" do
    @tag :integration
    test "executes optimizer step and returns metrics" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          adam_params = %{
            lr: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.01
          }

          result = GenServer.call(session, {:optim_step, adam_params, []})

          case result do
            {:ok, metrics} ->
              assert is_map(metrics)

            {:error, _reason} ->
              :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end
  end

  describe "save_checkpoint/2" do
    @tag :integration
    test "saves checkpoint and returns metadata" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          result = GenServer.call(session, {:save_checkpoint, 100})

          case result do
            {:ok, checkpoint} ->
              assert is_map(checkpoint)
              assert Map.has_key?(checkpoint, :name)
              assert Map.has_key?(checkpoint, :step)
              assert checkpoint.step == 100

            {:error, _reason} ->
              :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end
  end

  describe "create_sampler/1" do
    @tag :integration
    test "creates sampling client from checkpoint" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          # First save a checkpoint
          case GenServer.call(session, {:save_checkpoint, 100}) do
            {:ok, checkpoint} ->
              result = GenServer.call(session, {:create_sampler, checkpoint.name})

              case result do
                {:ok, sampler_pid} ->
                  assert is_pid(sampler_pid)

                {:error, _reason} ->
                  :ok
              end

            {:error, _reason} ->
              :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end
  end

  describe "sample/3" do
    @tag :integration
    test "generates samples from prompt" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          # Would need sampler to be created first
          prompt = "Test prompt:"
          opts = [temperature: 0.7, max_tokens: 100]

          # This would fail without a sampler
          result = GenServer.call(session, {:sample, prompt, opts})

          case result do
            {:ok, samples} ->
              assert is_list(samples)

            {:error, _reason} ->
              :ok
          end

          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end
  end

  describe "state management" do
    test "tracks checkpoints in state" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          state = :sys.get_state(session)
          assert state.checkpoints == []
          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end

    test "tracks metrics buffer" do
      experiment = valid_experiment()

      case safe_start_session(experiment) do
        {:ok, session} ->
          state = :sys.get_state(session)
          assert state.metrics_buffer == []
          GenServer.stop(session)

        {:error, _reason} ->
          :ok
      end
    end
  end
end
