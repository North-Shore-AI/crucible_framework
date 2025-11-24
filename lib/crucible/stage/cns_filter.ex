defmodule Crucible.Stage.CNSFilter do
  @moduledoc """
  Pipeline stage for filtering SNOs based on topological surrogate thresholds.

  This stage filters out SNOs that don't meet quality thresholds based on
  their topological surrogates (β₁ and fragility scores).

  ## Overview

  The stage:
  1. Reads SNOs with surrogate scores from context
  2. Applies configurable filters based on β₁ and fragility thresholds
  3. Removes or flags SNOs that don't meet criteria
  4. Logs filtered items for debugging
  5. Updates metrics with filter statistics

  ## Options

  - `:source` - Where to find SNOs (:examples or :outputs, default: :examples)
  - `:max_beta1` - Maximum allowed β₁ score (default: nil, no filtering)
  - `:max_fragility` - Maximum allowed fragility score (default: nil, no filtering)
  - `:min_beta1` - Minimum required β₁ score (default: nil, no filtering)
  - `:min_fragility` - Minimum required fragility score (default: nil, no filtering)
  - `:mode` - Filter mode (:remove or :flag, default: :remove)
  - `:log_filtered` - Whether to log filtered items (default: true)
  - `:reason_key` - Key to store filter reason when mode is :flag (default: :filter_reason)

  ## Filter Logic

  An SNO is filtered if ANY of these conditions are met:
  - β₁ score > max_beta1 (if specified)
  - fragility score > max_fragility (if specified)
  - β₁ score < min_beta1 (if specified)
  - fragility score < min_fragility (if specified)

  ## Example - Remove circular arguments

      stage = {Crucible.Stage.CNSFilter, %{
        source: :examples,
        max_beta1: 0,  # Remove any SNO with cycles
        mode: :remove
      }}

  ## Example - Flag high-fragility claims

      stage = {Crucible.Stage.CNSFilter, %{
        source: :outputs,
        max_fragility: 0.7,
        mode: :flag,
        reason_key: :quality_warning
      }}

  ## Output

  Updates the context with:
  - Filtered SNO list (when mode is :remove)
  - Flagged SNOs with reason (when mode is :flag)
  - `:cns_filter` metrics with filtering statistics
  """

  @behaviour Crucible.Stage

  alias Crucible.Context

  require Logger

  @impl true
  def run(%Context{} = ctx, opts) do
    opts = normalize_opts(opts)
    source = opts[:source] || :examples
    mode = opts[:mode] || :remove

    with {:ok, snos} <- extract_snos(ctx, source),
         {:ok, filtered, stats} <- apply_filters(snos, opts),
         {:ok, processed} <- process_by_mode(filtered, mode, opts),
         {:ok, updated_ctx} <- update_context(ctx, processed, stats, source) do
      {:ok, updated_ctx}
    else
      {:error, reason} = error ->
        Logger.error("[CNSFilter] Failed: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: "CNSFilter",
      description: "Filter SNOs based on topological surrogate thresholds",
      source: opts[:source] || :examples,
      mode: opts[:mode] || :remove,
      filters: build_filter_description(opts)
    }
  end

  # Private functions

  defp normalize_opts(opts) when is_map(opts), do: Map.to_list(opts)
  defp normalize_opts(opts) when is_list(opts), do: opts

  defp build_filter_description(opts) do
    filters = []

    filters =
      if opts[:max_beta1] do
        ["β₁ ≤ #{opts[:max_beta1]}" | filters]
      else
        filters
      end

    filters =
      if opts[:min_beta1] do
        ["β₁ ≥ #{opts[:min_beta1]}" | filters]
      else
        filters
      end

    filters =
      if opts[:max_fragility] do
        ["fragility ≤ #{opts[:max_fragility]}" | filters]
      else
        filters
      end

    filters =
      if opts[:min_fragility] do
        ["fragility ≥ #{opts[:min_fragility]}" | filters]
      else
        filters
      end

    case filters do
      [] -> "No filters configured"
      _ -> Enum.join(filters, ", ")
    end
  end

  defp extract_snos(%Context{} = ctx, source) do
    items =
      case source do
        :examples -> ctx.examples
        :outputs -> ctx.outputs
        _ -> nil
      end

    case items do
      nil ->
        {:error, {:invalid_source, source}}

      [] ->
        {:ok, []}

      items when is_list(items) ->
        # Check if items have surrogates
        if has_surrogates?(items) do
          {:ok, items}
        else
          {:error, :no_surrogates_found}
        end

      _ ->
        {:error, :invalid_items}
    end
  end

  defp has_surrogates?(items) do
    Enum.any?(items, fn item ->
      case item do
        %{surrogates: %{beta1: _, fragility: _}} -> true
        %{metadata: %{surrogates: %{beta1: _, fragility: _}}} -> true
        _ -> false
      end
    end)
  end

  defp apply_filters(snos, opts) do
    # Define filter criteria
    max_beta1 = opts[:max_beta1]
    min_beta1 = opts[:min_beta1]
    max_fragility = opts[:max_fragility]
    min_fragility = opts[:min_fragility]
    log_filtered = Keyword.get(opts, :log_filtered, true)

    # Track statistics
    stats = %{
      total: length(snos),
      filtered_by_beta1_max: 0,
      filtered_by_beta1_min: 0,
      filtered_by_fragility_max: 0,
      filtered_by_fragility_min: 0,
      passed: 0,
      filtered: 0
    }

    # Apply filters
    {filtered, final_stats} =
      Enum.reduce(snos, {[], stats}, fn sno, {acc, stats} ->
        surrogates = extract_surrogates(sno)

        cond do
          # Check max_beta1 threshold
          max_beta1 && surrogates.beta1 > max_beta1 ->
            if log_filtered do
              Logger.info(
                "[CNSFilter] Filtered SNO - β₁ (#{surrogates.beta1}) > max (#{max_beta1})"
              )
            end

            updated_stats = %{
              stats
              | filtered: stats.filtered + 1,
                filtered_by_beta1_max: stats.filtered_by_beta1_max + 1
            }

            {[{sno, :filtered, "β₁ > #{max_beta1}"} | acc], updated_stats}

          # Check min_beta1 threshold
          min_beta1 && surrogates.beta1 < min_beta1 ->
            if log_filtered do
              Logger.info(
                "[CNSFilter] Filtered SNO - β₁ (#{surrogates.beta1}) < min (#{min_beta1})"
              )
            end

            updated_stats = %{
              stats
              | filtered: stats.filtered + 1,
                filtered_by_beta1_min: stats.filtered_by_beta1_min + 1
            }

            {[{sno, :filtered, "β₁ < #{min_beta1}"} | acc], updated_stats}

          # Check max_fragility threshold
          max_fragility && surrogates.fragility > max_fragility ->
            if log_filtered do
              Logger.info(
                "[CNSFilter] Filtered SNO - fragility (#{surrogates.fragility}) > max (#{max_fragility})"
              )
            end

            updated_stats = %{
              stats
              | filtered: stats.filtered + 1,
                filtered_by_fragility_max: stats.filtered_by_fragility_max + 1
            }

            {[{sno, :filtered, "fragility > #{max_fragility}"} | acc], updated_stats}

          # Check min_fragility threshold
          min_fragility && surrogates.fragility < min_fragility ->
            if log_filtered do
              Logger.info(
                "[CNSFilter] Filtered SNO - fragility (#{surrogates.fragility}) < min (#{min_fragility})"
              )
            end

            updated_stats = %{
              stats
              | filtered: stats.filtered + 1,
                filtered_by_fragility_min: stats.filtered_by_fragility_min + 1
            }

            {[{sno, :filtered, "fragility < #{min_fragility}"} | acc], updated_stats}

          # Passed all filters
          true ->
            updated_stats = %{stats | passed: stats.passed + 1}
            {[{sno, :passed, nil} | acc], updated_stats}
        end
      end)

    # Reverse to maintain order
    {:ok, Enum.reverse(filtered), final_stats}
  end

  defp extract_surrogates(item) do
    case item do
      %{surrogates: surrogates} ->
        surrogates

      %{metadata: %{surrogates: surrogates}} ->
        surrogates

      _ ->
        %{beta1: 0, fragility: 0.0}
    end
  end

  defp process_by_mode(filtered_results, :remove, _opts) do
    # Only keep items that passed
    passed =
      filtered_results
      |> Enum.filter(fn {_sno, status, _reason} -> status == :passed end)
      |> Enum.map(fn {sno, _status, _reason} -> sno end)

    {:ok, passed}
  end

  defp process_by_mode(filtered_results, :flag, opts) do
    # Keep all items but add filter reason to filtered ones
    reason_key = opts[:reason_key] || :filter_reason

    flagged =
      Enum.map(filtered_results, fn {sno, status, reason} ->
        case status do
          :passed ->
            sno

          :filtered ->
            # Add filter reason to item
            case sno do
              %{metadata: meta} when is_map(meta) ->
                %{sno | metadata: Map.put(meta, reason_key, reason)}

              _ ->
                Map.put(sno, reason_key, reason)
            end
        end
      end)

    {:ok, flagged}
  end

  defp process_by_mode(_filtered_results, mode, _opts) do
    {:error, {:invalid_mode, mode}}
  end

  defp update_context(ctx, processed_snos, stats, source) do
    # Update the source in context
    updated_ctx =
      case source do
        :examples ->
          %{ctx | examples: processed_snos}

        :outputs ->
          %{ctx | outputs: processed_snos}
      end

    # Add metrics
    filter_metrics = %{
      stats: stats,
      filter_rate: Float.round(stats.filtered / max(stats.total, 1), 4),
      pass_rate: Float.round(stats.passed / max(stats.total, 1), 4)
    }

    updated_ctx = %{
      updated_ctx
      | metrics: Map.put(updated_ctx.metrics, :cns_filter, filter_metrics)
    }

    # Add to assigns
    assigns = updated_ctx.assigns || %{}

    updated_assigns =
      Map.put(assigns, :cns_filter, %{
        total_filtered: stats.filtered,
        total_passed: stats.passed,
        filter_stats: stats
      })

    updated_ctx = %{updated_ctx | assigns: updated_assigns}

    # Log summary
    Logger.info(
      "[CNSFilter] Processed #{stats.total} items: #{stats.passed} passed, #{stats.filtered} filtered"
    )

    {:ok, updated_ctx}
  end
end
