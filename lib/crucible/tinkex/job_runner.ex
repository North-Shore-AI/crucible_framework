defmodule Crucible.Tinkex.JobRunner do
  @moduledoc """
  Executes `Crucible.Tinkex.Job` manifests using the Tinkex SDK (when enabled)
  or a simulated runner (default).

  This module is the only place that touches Tinkex credentials, satisfying the
  CNS single-owner requirement. External callers submit jobs via the public API
  and never supply credentials; the runner pulls them from application config.
  """

  alias Crucible.Tinkex.Config
  alias Crucible.Tinkex.Job
  alias Crucible.Tinkex.Telemetry

  @type result :: :ok | {:error, term()}

  @spec submit(Job.t()) :: result()
  def submit(%Job{} = job) do
    artifacts_path = ensure_artifacts_dir(job)
    write_manifest(job, artifacts_path)

    case runner_mode() do
      :tinkex -> run_with_tinkex(job, artifacts_path)
      _ -> simulate(job, artifacts_path)
    end
  end

  defp runner_mode do
    Application.get_env(:crucible_framework, :runner_mode, :simulate)
  end

  defp ensure_artifacts_dir(%Job{artifacts_path: path}) do
    File.mkdir_p!(path)
    File.mkdir_p!(Path.join(path, "checkpoints"))
    path
  end

  defp write_manifest(%Job{} = job, artifacts_path) do
    manifest = %{
      job_id: job.id,
      name: job.name,
      status: job.status,
      spec: job.spec,
      inserted_at: job.inserted_at,
      updated_at: job.updated_at
    }

    File.write!(Path.join(artifacts_path, "manifest.json"), Jason.encode!(manifest, pretty: true))
  end

  defp simulate(%Job{} = job, _artifacts_path) do
    # Minimal simulation to keep the API surface usable without hitting Tinkex.
    for step <- 1..3 do
      Telemetry.emit_training_step(%{
        job_id: job.id,
        step: step,
        epoch: 1,
        loss: Float.round(1.0 - step * 0.1, 3),
        citation_invalid_rate: 0.0
      })
    end

    Telemetry.emit_checkpoint_saved(%{
      job_id: job.id,
      name: "#{job.id}_sim_checkpoint",
      step: 3
    })

    Telemetry.emit_evaluation_complete(%{
      job_id: job.id,
      adapter_name: "simulated",
      samples: 0,
      metrics: %{
        schema_compliance: 0.0,
        citation_accuracy: 0.0,
        mean_entailment: 0.0,
        overall_pass_rate: 0.0
      }
    })

    :ok
  end

  defp run_with_tinkex(%Job{} = job, artifacts_path) do
    with {:ok, config} <- load_config(),
         {:ok, service} <- start_service(config),
         {:ok, training_client} <- create_training_client(service, config, job.spec) do
      # Training loop wiring is backend-specific; placeholder keeps credentials bounded here.
      _ = maybe_emit_job_start(job)

      # TODO: integrate dataset manifest ingestion and real forward/backward calls.
      _ = training_client
      _ = artifacts_path

      :ok
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, :tinkex_failed}
    end
  end

  defp load_config do
    config = Config.new()

    case Config.validate(config) do
      :ok -> {:ok, config}
      {:error, reason} -> {:error, reason}
    end
  end

  defp start_service(config) do
    opts = [config: Config.to_tinkex_opts(config)]

    case safe_apply(Tinkex.ServiceClient, :start_link, [opts]) do
      {:ok, pid} -> {:ok, pid}
      other -> other
    end
  end

  defp create_training_client(service, config, spec) do
    base_model =
      Map.get(spec, :base_model) || Map.get(spec, "base_model") || config.default_base_model

    rank = Map.get(spec, :rank) || Map.get(spec, "rank") || config.default_lora_rank

    apply_opts = [service, [base_model: base_model, rank: rank]]
    safe_apply(Tinkex.ServiceClient, :create_lora_training_client, apply_opts)
  end

  defp maybe_emit_job_start(job) do
    Telemetry.emit_training_step(%{
      job_id: job.id,
      step: 0,
      epoch: 0,
      loss: nil,
      citation_invalid_rate: 0.0
    })
  end

  defp safe_apply(mod, fun, args) do
    if Code.ensure_loaded?(mod) and function_exported?(mod, fun, length(args)) do
      try do
        apply(mod, fun, args)
      rescue
        exception -> {:error, exception}
      end
    else
      {:error, :not_available}
    end
  end
end
