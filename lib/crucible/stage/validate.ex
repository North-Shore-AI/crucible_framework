defmodule Crucible.Stage.Validate do
  @moduledoc """
  Pre-flight validation stage for experiments.

  This stage validates the experiment configuration before execution begins,
  catching configuration errors early and providing clear feedback.

  ## Validation Checks

  1. **Backend Validation**
     - Backend ID is registered in application config
     - Backend module exists and is loadable

  2. **Pipeline Stage Validation**
     - All stage names resolve to modules
     - Stage modules implement the Crucible.Stage behaviour
     - No duplicate stage names (unless intentional)

  3. **Dataset Validation**
     - Dataset provider module exists (if specified)
     - Dataset configuration is valid

  4. **Reliability Configuration**
     - Ensemble members are valid (if configured)
     - Hedging configuration is valid (if configured)
     - Statistical test configuration is valid

  5. **Output Validation**
     - Output sinks are valid
     - Output formats are supported

  ## Usage

  Add as the first stage in your pipeline:

      pipeline: [
        %StageDef{name: :validate},
        %StageDef{name: :data_load},
        # ... rest of pipeline
      ]

  ## Configuration

  Validation can be configured via stage options:

      %StageDef{
        name: :validate,
        options: %{
          strict: true,              # Fail on warnings (default: false)
          skip_backend: false,       # Skip backend validation (default: false)
          skip_dataset: false,       # Skip dataset validation (default: false)
          allow_missing_adapters: true  # Allow missing analysis adapters (default: true)
        }
      }

  ## Examples

      # Basic validation
      pipeline: [%StageDef{name: :validate}, ...]

      # Strict validation (warnings are errors)
      pipeline: [%StageDef{name: :validate, options: %{strict: true}}, ...]

      # Skip certain validations
      pipeline: [
        %StageDef{
          name: :validate,
          options: %{skip_backend: true}
        },
        ...
      ]
  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.{Context, Registry}
  alias CrucibleIR.{DatasetRef, BackendRef}
  alias CrucibleIR.Reliability.{Config, Ensemble}

  @impl true
  def run(%Context{experiment: experiment} = ctx, opts) do
    Logger.info("Validating experiment: #{experiment.id}")

    # Collect validation results
    validation_result = %{
      backend: validate_backend(experiment.backend, opts),
      stages: validate_stages(experiment.pipeline, opts),
      dataset: validate_dataset(experiment.dataset, opts),
      reliability: validate_reliability(experiment.reliability, opts),
      outputs: validate_outputs(experiment.outputs, opts)
    }

    # Check for errors
    errors = collect_errors(validation_result)
    warnings = collect_warnings(validation_result)

    strict = Map.get(opts, :strict, false)

    cond do
      errors != [] ->
        # Validation failed with errors
        Logger.error("Experiment validation failed with #{length(errors)} errors")
        Enum.each(errors, &Logger.error("  - #{&1}"))
        {:error, {:validation_failed, errors}}

      strict && warnings != [] ->
        # Strict mode: warnings are errors
        Logger.error(
          "Experiment validation failed (strict mode) with #{length(warnings)} warnings"
        )

        Enum.each(warnings, &Logger.warning("  - #{&1}"))
        {:error, {:validation_failed, warnings}}

      warnings != [] ->
        # Validation passed with warnings
        Logger.warning("Experiment validation passed with #{length(warnings)} warnings")
        Enum.each(warnings, &Logger.warning("  - #{&1}"))

        new_ctx =
          ctx
          |> Context.put_metric(:validation, %{
            status: :passed_with_warnings,
            warnings: warnings,
            details: validation_result
          })

        {:ok, new_ctx}

      true ->
        # Validation passed
        Logger.info("Experiment validation passed")

        new_ctx =
          ctx
          |> Context.put_metric(:validation, %{
            status: :passed,
            details: validation_result
          })

        {:ok, new_ctx}
    end
  end

  @impl true
  def describe(_opts) do
    %{
      stage: :validate,
      description: "Pre-flight validation of experiment configuration",
      checks: [:backend, :stages, :dataset, :reliability, :outputs]
    }
  end

  # ============================================================================
  # Backend Validation
  # ============================================================================

  defp validate_backend(_backend, %{skip_backend: true}), do: %{status: :skipped}

  defp validate_backend(nil, _opts) do
    %{
      status: :error,
      errors: ["No backend configured"]
    }
  end

  defp validate_backend(%BackendRef{id: id} = ref, _opts) do
    errors = []
    warnings = []

    # Check if backend is registered
    {errors, warnings} =
      case Registry.backend_module(id) do
        {:ok, module} ->
          # Backend found, check if module loads
          if Code.ensure_loaded?(module) do
            {errors, warnings}
          else
            {["Backend module #{inspect(module)} cannot be loaded" | errors], warnings}
          end

        {:error, {:unknown_backend, ^id}} ->
          {["Backend :#{id} is not registered in application config" | errors], warnings}

        {:error, :no_backends_configured} ->
          {["No backends configured in application config" | errors], warnings}
      end

    # Validate backend options
    {errors, warnings} =
      case ref.options do
        nil ->
          {errors, ["Backend has no options configured (may use defaults)" | warnings]}

        %{} = opts when map_size(opts) == 0 ->
          {errors, ["Backend has empty options map (may use defaults)" | warnings]}

        %{} ->
          {errors, warnings}
      end

    case {errors, warnings} do
      {[], []} -> %{status: :ok, backend_id: id}
      {[], warnings} -> %{status: :warning, backend_id: id, warnings: warnings}
      {errors, _} -> %{status: :error, backend_id: id, errors: errors}
    end
  end

  # ============================================================================
  # Stage Validation
  # ============================================================================

  defp validate_stages(pipeline, opts) when is_list(pipeline) do
    results =
      Enum.map(pipeline, fn stage_def ->
        validate_single_stage(stage_def, opts)
      end)

    errors = Enum.flat_map(results, &Map.get(&1, :errors, []))
    warnings = Enum.flat_map(results, &Map.get(&1, :warnings, []))

    # Check for duplicate stages
    stage_names = Enum.map(pipeline, & &1.name)
    duplicates = stage_names -- Enum.uniq(stage_names)

    warnings =
      if duplicates != [] do
        ["Duplicate stages found: #{inspect(Enum.uniq(duplicates))}" | warnings]
      else
        warnings
      end

    case {errors, warnings} do
      {[], []} -> %{status: :ok, stage_count: length(pipeline)}
      {[], warnings} -> %{status: :warning, stage_count: length(pipeline), warnings: warnings}
      {errors, _} -> %{status: :error, stage_count: length(pipeline), errors: errors}
    end
  end

  defp validate_stages(_, _opts) do
    %{status: :error, errors: ["Pipeline is not a list"]}
  end

  defp validate_single_stage(stage_def, _opts) do
    errors = []
    warnings = []

    # Validate stage name resolution
    {errors, warnings} =
      case stage_def do
        %{module: mod} when not is_nil(mod) ->
          # Module explicitly specified
          if Code.ensure_loaded?(mod) do
            if function_exported?(mod, :run, 2) do
              {errors, warnings}
            else
              {["Stage module #{inspect(mod)} does not implement run/2" | errors], warnings}
            end
          else
            {["Stage module #{inspect(mod)} cannot be loaded" | errors], warnings}
          end

        %{name: name} ->
          # Resolve from registry
          case Registry.stage_module(name) do
            {:ok, module} ->
              if Code.ensure_loaded?(module) do
                {errors, warnings}
              else
                {[
                   "Stage :#{name} resolves to #{inspect(module)} which cannot be loaded" | errors
                 ], warnings}
              end

            {:error, {:unknown_stage, ^name}} ->
              {["Stage :#{name} is not registered" | errors], warnings}

            {:error, :no_stage_registry} ->
              {["No stage registry configured" | errors], warnings}
          end

        _ ->
          {["Stage definition missing both :name and :module" | errors], warnings}
      end

    case {errors, warnings} do
      {[], []} -> %{status: :ok, stage: stage_def.name}
      {[], warnings} -> %{status: :warning, stage: stage_def.name, warnings: warnings}
      {errors, _} -> %{status: :error, stage: stage_def.name, errors: errors}
    end
  end

  # ============================================================================
  # Dataset Validation
  # ============================================================================

  defp validate_dataset(_dataset, %{skip_dataset: true}), do: %{status: :skipped}
  defp validate_dataset(nil, _opts), do: %{status: :ok, message: "No dataset configured"}

  defp validate_dataset(%DatasetRef{provider: nil}, _opts) do
    %{
      status: :warning,
      warnings: ["Dataset has no provider specified (will use InMemory provider)"]
    }
  end

  defp validate_dataset(%DatasetRef{provider: provider} = ref, _opts) do
    errors = []
    warnings = []

    # Check if provider module exists
    {errors, warnings} =
      if is_atom(provider) && Code.ensure_loaded?(provider) do
        {errors, warnings}
      else
        {["Dataset provider #{inspect(provider)} cannot be loaded" | errors], warnings}
      end

    # Check dataset name
    {errors, warnings} =
      case ref.name do
        nil -> {errors, ["Dataset has no name specified" | warnings]}
        _ -> {errors, warnings}
      end

    case {errors, warnings} do
      {[], []} -> %{status: :ok, provider: provider}
      {[], warnings} -> %{status: :warning, provider: provider, warnings: warnings}
      {errors, _} -> %{status: :error, provider: provider, errors: errors}
    end
  end

  defp validate_dataset(datasets, opts) when is_list(datasets) do
    results = Enum.map(datasets, &validate_dataset(&1, opts))
    errors = Enum.flat_map(results, &Map.get(&1, :errors, []))
    warnings = Enum.flat_map(results, &Map.get(&1, :warnings, []))

    case {errors, warnings} do
      {[], []} -> %{status: :ok, dataset_count: length(datasets)}
      {[], warnings} -> %{status: :warning, dataset_count: length(datasets), warnings: warnings}
      {errors, _} -> %{status: :error, dataset_count: length(datasets), errors: errors}
    end
  end

  # ============================================================================
  # Reliability Configuration Validation
  # ============================================================================

  defp validate_reliability(%Config{} = config, opts) do
    ensemble_result = validate_ensemble(config.ensemble, opts)
    hedging_result = validate_hedging(config.hedging, opts)
    stats_result = validate_stats(config.stats, opts)

    errors =
      [ensemble_result, hedging_result, stats_result]
      |> Enum.flat_map(&Map.get(&1, :errors, []))

    warnings =
      [ensemble_result, hedging_result, stats_result]
      |> Enum.flat_map(&Map.get(&1, :warnings, []))

    case {errors, warnings} do
      {[], []} -> %{status: :ok}
      {[], warnings} -> %{status: :warning, warnings: warnings}
      {errors, _} -> %{status: :error, errors: errors}
    end
  end

  defp validate_reliability(nil, _opts) do
    %{status: :ok, message: "No reliability config specified"}
  end

  defp validate_ensemble(%Ensemble{strategy: :none}, _opts) do
    %{status: :ok, message: "Ensemble disabled"}
  end

  defp validate_ensemble(%Ensemble{strategy: strategy, models: models}, _opts)
       when is_list(models) do
    errors = []
    warnings = []

    # Check if strategy is valid
    valid_strategies = [:majority_vote, :weighted_vote, :best_confidence, :unanimous]

    {errors, warnings} =
      if strategy in valid_strategies do
        {errors, warnings}
      else
        {["Unknown ensemble strategy: #{inspect(strategy)}" | errors], warnings}
      end

    # Check if models are valid
    {errors, warnings} =
      if models == [] do
        {["Ensemble has no models configured" | errors], warnings}
      else
        # Validate each model
        model_errors =
          Enum.flat_map(models, fn
            %BackendRef{id: id} ->
              case Registry.backend_module(id) do
                {:ok, _} -> []
                {:error, _} -> ["Ensemble model backend :#{id} is not registered"]
              end

            _ ->
              ["Invalid ensemble model (must be BackendRef)"]
          end)

        {model_errors ++ errors, warnings}
      end

    case {errors, warnings} do
      {[], []} -> %{status: :ok, strategy: strategy, model_count: length(models)}
      {[], warnings} -> %{status: :warning, strategy: strategy, warnings: warnings}
      {errors, _} -> %{status: :error, strategy: strategy, errors: errors}
    end
  end

  defp validate_ensemble(nil, _opts) do
    %{status: :ok, message: "No ensemble config"}
  end

  defp validate_hedging(%{strategy: :off}, _opts) do
    %{status: :ok, message: "Hedging disabled"}
  end

  defp validate_hedging(%{strategy: strategy}, _opts) do
    valid_strategies = [:fixed_delay, :percentile, :adaptive, :exponential_backoff, :off]

    if strategy in valid_strategies do
      %{status: :ok, strategy: strategy}
    else
      %{status: :error, errors: ["Unknown hedging strategy: #{inspect(strategy)}"]}
    end
  end

  defp validate_hedging(nil, _opts) do
    %{status: :ok, message: "No hedging config"}
  end

  defp validate_stats(%{tests: tests}, _opts) when is_list(tests) do
    valid_tests = [
      :ttest,
      :welch_ttest,
      :paired_ttest,
      :wilcoxon,
      :mann_whitney,
      :bootstrap,
      :anova,
      :kruskal_wallis
    ]

    invalid_tests = tests -- valid_tests

    if invalid_tests == [] do
      %{status: :ok, test_count: length(tests)}
    else
      %{status: :error, errors: ["Unknown statistical tests: #{inspect(invalid_tests)}"]}
    end
  end

  defp validate_stats(nil, _opts) do
    %{status: :ok, message: "No stats config"}
  end

  # ============================================================================
  # Output Validation
  # ============================================================================

  defp validate_outputs(outputs, _opts) when is_list(outputs) do
    if outputs == [] do
      %{status: :warning, warnings: ["No outputs configured"]}
    else
      errors =
        Enum.flat_map(outputs, fn output ->
          case output do
            %{name: nil} ->
              ["Output missing name"]

            %{formats: []} ->
              ["Output has no formats specified"]

            %{sink: sink} when sink not in [:file, :memory, :database] ->
              ["Unknown sink: #{sink}"]

            _ ->
              []
          end
        end)

      if errors == [] do
        %{status: :ok, output_count: length(outputs)}
      else
        %{status: :error, errors: errors}
      end
    end
  end

  defp validate_outputs(_, _opts) do
    %{status: :ok, message: "No outputs configured"}
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp collect_errors(validation_result) do
    validation_result
    |> Map.values()
    |> Enum.flat_map(fn
      %{errors: errors} -> errors
      _ -> []
    end)
  end

  defp collect_warnings(validation_result) do
    validation_result
    |> Map.values()
    |> Enum.flat_map(fn
      %{warnings: warnings} -> warnings
      _ -> []
    end)
  end
end
