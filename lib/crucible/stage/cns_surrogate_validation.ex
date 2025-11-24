defmodule Crucible.Stage.CNSSurrogateValidation do
  @moduledoc """
  Pipeline stage for computing CNS topological surrogates on SNO-like data.

  This stage computes lightweight topological surrogates (β₁ and fragility)
  for validating the topology-logic correlation hypothesis before investing
  in full TDA infrastructure.

  ## Overview

  The stage:
  1. Extracts SNOs from the context (examples or outputs)
  2. Computes β₁ surrogate from causal link graphs
  3. Computes fragility surrogate from embedding variance
  4. Adds surrogate scores to each SNO's metadata
  5. Updates context metrics with aggregate statistics

  ## Options

  - `:source` - Where to find SNOs (:examples or :outputs, default: :examples)
  - `:k` - Number of neighbors for fragility computation (default: 5)
  - `:metric` - Distance metric for fragility (:cosine or :euclidean, default: :cosine)
  - `:validate` - Whether to validate correlation if labels exist (default: false)
  - `:tag_snos` - Whether to add surrogate scores to SNO metadata (default: true)

  ## Expected Context

  The context should contain either:
  - `examples`: List of SNO-like maps with :causal_links and :embeddings
  - `outputs`: Model outputs with similar structure

  ## Output

  Updates the context with:
  - Modified SNOs with :surrogates in metadata
  - `:cns_surrogates` metrics with aggregate statistics

  ## Example

      stage = {Crucible.Stage.CNSSurrogateValidation, %{
        source: :examples,
        k: 5,
        metric: :cosine,
        validate: true
      }}

  ## Gate 1 Validation

  When `:validate` is true and labels are present, computes correlation
  metrics to determine if surrogates meet Gate 1 criteria (r > 0.35).
  """

  @behaviour Crucible.Stage
  @compile {:no_warn_undefined, CNS.Topology.Surrogates}

  alias Crucible.Context

  require Logger

  defmodule FallbackSurrogate do
    def compute_surrogates(_sno, _opts \\ []), do: %{beta1: 0, fragility: 0.0}

    def validate_correlation(_snos, _opts \\ []) do
      %{
        beta1_correlation: 0.0,
        beta1_p_value: 1.0,
        fragility_correlation: 0.0,
        fragility_p_value: 1.0,
        combined_correlation: 0.0,
        combined_p_value: 1.0,
        n_samples: 0,
        passes_gate1: false
      }
    end
  end

  @impl true
  def run(%Context{} = ctx, opts) do
    opts = normalize_opts(opts)
    source = opts[:source]

    with {:ok, snos} <- extract_snos(ctx, source),
         {:ok, processed} <- process_snos(snos, opts),
         {:ok, metrics} <- compute_metrics(processed, opts),
         {:ok, updated_ctx} <- update_context(ctx, processed, metrics, source) do
      {:ok, updated_ctx}
    else
      {:error, reason} = error ->
        Logger.error("[CNSSurrogateValidation] Failed: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def describe(opts) do
    %{
      stage: "CNSSurrogateValidation",
      description: "Compute topological surrogates for SNOs",
      source: opts[:source] || :examples,
      k: opts[:k] || 5,
      metric: opts[:metric] || :cosine,
      validate: opts[:validate] || false
    }
  end

  # Private functions

  defp normalize_opts(opts) when is_map(opts), do: Map.to_list(opts)
  defp normalize_opts(opts) when is_list(opts), do: opts

  defp extract_snos(%Context{} = ctx, :examples) do
    case ctx.examples do
      nil ->
        {:error, :no_examples}

      [] ->
        {:ok, []}

      examples when is_list(examples) ->
        # Ensure examples have the required structure
        prepared = prepare_snos(examples)
        {:ok, prepared}

      _ ->
        {:error, :invalid_examples}
    end
  end

  defp extract_snos(%Context{} = ctx, :outputs) do
    case ctx.outputs do
      nil ->
        {:error, :no_outputs}

      [] ->
        {:ok, []}

      outputs when is_list(outputs) ->
        prepared = prepare_snos(outputs)
        {:ok, prepared}

      _ ->
        {:error, :invalid_outputs}
    end
  end

  defp extract_snos(_ctx, source) do
    {:error, {:invalid_source, source}}
  end

  defp prepare_snos(items) do
    Enum.map(items, fn item ->
      # Extract causal links - handle various formats
      causal_links =
        case item do
          %{causal_links: links} when is_list(links) ->
            links

          %{"causal_links" => links} when is_list(links) ->
            links

          %{relations: relations} when is_list(relations) ->
            # Convert relations to causal links
            extract_causal_links_from_relations(relations)

          _ ->
            []
        end

      # Extract embeddings
      embeddings =
        case item do
          %{embeddings: emb} when is_list(emb) ->
            emb

          %{"embeddings" => emb} when is_list(emb) ->
            emb

          %{embedding: emb} when is_list(emb) ->
            # Single embedding
            [emb]

          %{"embedding" => emb} when is_list(emb) ->
            [emb]

          _ ->
            []
        end

      # Extract label if present (for validation)
      label =
        case item do
          %{label: l} -> l
          %{"label" => l} -> l
          %{is_circular: true} -> 1
          %{is_circular: false} -> 0
          _ -> nil
        end

      %{
        original: item,
        causal_links: causal_links,
        embeddings: embeddings,
        label: label
      }
    end)
  end

  defp extract_causal_links_from_relations(relations) do
    # Extract causal links from relation format like "c2 supports c1"
    relations
    |> Enum.filter(fn rel -> match?({:ok, _, _, _}, parse_relation(rel)) end)
    |> Enum.map(fn rel ->
      case parse_relation(rel) do
        {:ok, source, _rel_type, target} -> {source, target}
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_relation(relation) when is_binary(relation) do
    case String.split(relation, " ") do
      [source, rel_type, target] ->
        {:ok, source, rel_type, target}

      _ ->
        {:error, :invalid_format}
    end
  end

  defp parse_relation(%{source: s, relation: rel_type, target: t}) do
    {:ok, s, rel_type, t}
  end

  defp parse_relation(_), do: {:error, :invalid_format}

  defp process_snos(snos, opts) do
    processed =
      Enum.map(snos, fn sno ->
        surrogates = compute_surrogates(sno, opts)

        # Add surrogates to the SNO
        %{sno | surrogates: surrogates}
      end)

    {:ok, processed}
  end

  defp compute_surrogates(sno, opts) do
    surrogate_mod = surrogate_module()

    if function_exported?(surrogate_mod, :compute_surrogates, 2) do
      surrogate_mod.compute_surrogates(sno, opts)
    else
      # Fallback inline implementation
      %{
        beta1: compute_beta1_fallback(sno.causal_links),
        fragility: compute_fragility_fallback(sno.embeddings, opts)
      }
    end
  end

  defp compute_beta1_fallback([]), do: 0

  defp compute_beta1_fallback(links) do
    # Build graph from links
    graph =
      Enum.reduce(links, %{}, fn {source, target}, acc ->
        acc
        |> Map.update(source, [target], &[target | &1])
        |> Map.put_new(target, [])
      end)

    # Simple cycle detection
    if has_cycle?(graph), do: 1, else: 0
  end

  defp has_cycle?(graph) when map_size(graph) == 0, do: false

  defp has_cycle?(graph) do
    nodes = Map.keys(graph)
    visited = MapSet.new()
    rec_stack = MapSet.new()

    Enum.any?(nodes, fn node ->
      if not MapSet.member?(visited, node) do
        detect_cycle_dfs(node, graph, visited, rec_stack)
      else
        false
      end
    end)
  end

  defp detect_cycle_dfs(node, graph, visited, rec_stack) do
    visited = MapSet.put(visited, node)
    rec_stack = MapSet.put(rec_stack, node)

    children = Map.get(graph, node, [])

    has_cycle =
      Enum.any?(children, fn child ->
        cond do
          not MapSet.member?(visited, child) ->
            detect_cycle_dfs(child, graph, visited, rec_stack)

          MapSet.member?(rec_stack, child) ->
            true

          true ->
            false
        end
      end)

    has_cycle
  end

  defp compute_fragility_fallback([], _opts), do: 0.0

  defp compute_fragility_fallback([_], _opts), do: 0.0

  defp compute_fragility_fallback(embeddings, _opts) when is_list(embeddings) do
    # Compute pairwise distances
    n = length(embeddings)

    distances =
      for i <- 0..(n - 2), j <- (i + 1)..(n - 1) do
        emb1 = Enum.at(embeddings, i)
        emb2 = Enum.at(embeddings, j)
        euclidean_distance(emb1, emb2)
      end

    if distances == [] do
      0.0
    else
      # Compute variance and normalize
      mean = Enum.sum(distances) / length(distances)

      variance =
        distances
        |> Enum.map(fn d -> (d - mean) * (d - mean) end)
        |> Enum.sum()
        |> Kernel./(length(distances))

      # Normalize using tanh
      normalized = :math.tanh(variance * 0.5)
      Float.round(normalized, 4)
    end
  end

  defp euclidean_distance(v1, v2) do
    v1
    |> Enum.zip(v2)
    |> Enum.map(fn {a, b} -> (a - b) * (a - b) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp compute_metrics(processed_snos, opts) do
    # Compute aggregate statistics
    beta1_values = Enum.map(processed_snos, & &1.surrogates.beta1)
    fragility_values = Enum.map(processed_snos, & &1.surrogates.fragility)

    metrics = %{
      count: length(processed_snos),
      beta1: %{
        mean: mean(beta1_values),
        max: Enum.max(beta1_values, fn -> 0 end),
        min: Enum.min(beta1_values, fn -> 0 end),
        cycles_detected: Enum.count(beta1_values, &(&1 > 0))
      },
      fragility: %{
        mean: mean(fragility_values),
        max: Enum.max(fragility_values, fn -> 0.0 end),
        min: Enum.min(fragility_values, fn -> 0.0 end),
        high_fragility_count: Enum.count(fragility_values, &(&1 > 0.5))
      }
    }

    # Add validation metrics if requested
    metrics =
      if opts[:validate] && has_labels?(processed_snos) do
        validation = validate_correlation(processed_snos)
        Map.put(metrics, :validation, validation)
      else
        metrics
      end

    {:ok, metrics}
  end

  defp has_labels?(snos) do
    Enum.all?(snos, fn sno -> sno.label != nil end)
  end

  defp validate_correlation(snos) do
    surrogate_mod = surrogate_module()

    if function_exported?(surrogate_mod, :validate_correlation, 2) do
      surrogate_mod.validate_correlation(snos, [])
    else
      # Fallback simple correlation
      beta1_values = Enum.map(snos, & &1.surrogates.beta1)
      labels = Enum.map(snos, & &1.label)

      correlation = simple_correlation(beta1_values, labels)

      %{
        beta1_correlation: correlation,
        passes_gate1: correlation > 0.35,
        note: "Using fallback correlation"
      }
    end
  end

  defp surrogate_module do
    if Code.ensure_loaded?(CNS.Topology.Surrogates) do
      CNS.Topology.Surrogates
    else
      FallbackSurrogate
    end
  end

  defp simple_correlation(x_values, y_values) do
    n = length(x_values)

    if n < 2 do
      0.0
    else
      x_mean = Enum.sum(x_values) / n
      y_mean = Enum.sum(y_values) / n

      numerator =
        Enum.zip(x_values, y_values)
        |> Enum.map(fn {x, y} -> (x - x_mean) * (y - y_mean) end)
        |> Enum.sum()

      x_std =
        :math.sqrt(Enum.sum(Enum.map(x_values, fn x -> (x - x_mean) * (x - x_mean) end)) / n)

      y_std =
        :math.sqrt(Enum.sum(Enum.map(y_values, fn y -> (y - y_mean) * (y - y_mean) end)) / n)

      if x_std == 0 or y_std == 0 do
        0.0
      else
        Float.round(numerator / (n * x_std * y_std), 4)
      end
    end
  end

  defp mean([]), do: 0.0

  defp mean(values) do
    Float.round(Enum.sum(values) / length(values), 4)
  end

  defp update_context(ctx, processed_snos, metrics, source) do
    # Add surrogates to original items if tag_snos is true
    updated_items =
      Enum.map(processed_snos, fn sno ->
        original = sno.original

        # Add surrogates to metadata
        updated =
          case original do
            %{metadata: meta} when is_map(meta) ->
              %{original | metadata: Map.put(meta, :surrogates, sno.surrogates)}

            _ ->
              Map.put(original, :surrogates, sno.surrogates)
          end

        updated
      end)

    # Update context
    updated_ctx =
      case source do
        :examples ->
          %{ctx | examples: updated_items}

        :outputs ->
          %{ctx | outputs: updated_items}
      end

    # Add metrics
    updated_ctx = %{
      updated_ctx
      | metrics: Map.put(updated_ctx.metrics, :cns_surrogates, metrics)
    }

    # Add to assigns for downstream stages
    updated_ctx = %{
      updated_ctx
      | assigns:
          Map.put(updated_ctx.assigns || %{}, :cns_surrogates, %{
            processed: length(processed_snos),
            metrics: metrics
          })
    }

    {:ok, updated_ctx}
  end
end
