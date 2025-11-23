defmodule Crucible.Thinker.Utils do
  @moduledoc """
  Utility functions for Thinker experiments.
  """

  @doc """
  Generates a unique ID for experiments and sessions.

  Returns a 16-character hex string.
  """
  @spec generate_id() :: String.t()
  def generate_id do
    :crypto.strong_rand_bytes(8)
    |> Base.encode16(case: :lower)
  end

  @doc """
  Calculates aggregate metrics from training results.
  """
  @spec calculate_metrics([map()]) :: map()
  def calculate_metrics([]) do
    %{
      mean_loss: nil,
      loss_reduction: nil,
      mean_citation_invalid_rate: nil,
      total_steps: 0
    }
  end

  def calculate_metrics(results) when is_list(results) do
    losses = Enum.map(results, & &1.loss)
    citation_rates = Enum.map(results, &(&1[:citation_invalid_rate] || 0.0))

    mean_loss = Enum.sum(losses) / length(losses)
    first_loss = hd(losses)
    last_loss = List.last(losses)
    loss_reduction = first_loss - last_loss

    mean_citation_rate = Enum.sum(citation_rates) / length(citation_rates)

    %{
      mean_loss: mean_loss,
      loss_reduction: loss_reduction,
      mean_citation_invalid_rate: mean_citation_rate,
      total_steps: length(results),
      min_loss: Enum.min(losses),
      max_loss: Enum.max(losses),
      final_loss: last_loss
    }
  end

  @doc """
  Validates evaluation results against quality targets.
  """
  @spec validate_quality(map(), map()) :: map()
  def validate_quality(results, targets) do
    assessments =
      Enum.map(targets, fn {metric, target} ->
        actual = Map.get(results, metric, 0)
        passed = actual >= target

        %{
          metric: metric,
          target: target,
          actual: actual,
          passed: passed,
          delta: Float.round(actual - target, 3)
        }
      end)

    all_passed = Enum.all?(assessments, & &1.passed)
    passed_count = Enum.count(assessments, & &1.passed)

    %{
      passed: all_passed,
      details: assessments,
      summary: "#{passed_count}/#{length(assessments)} quality targets met"
    }
  end

  @doc """
  Generates a checkpoint name for the current training state.
  """
  @spec checkpoint_name(String.t(), pos_integer()) :: String.t()
  def checkpoint_name(experiment_id, step) do
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    "#{experiment_id}_step_#{step}_#{timestamp}"
  end

  @doc """
  Creates sampling parameters map.
  """
  @spec sampling_params(keyword()) :: map()
  def sampling_params(opts \\ []) do
    %{
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.95),
      max_tokens: Keyword.get(opts, :max_tokens, 512),
      stop_sequences: Keyword.get(opts, :stop_sequences, [])
    }
  end

  @doc """
  Builds a Tinkex Datum from a training sample.

  Tokenizes input and output, creates proper token arrays.
  """
  @spec build_datum(map(), String.t(), keyword()) :: Tinkex.Types.Datum.t()
  def build_datum(sample, model_name, opts \\ []) do
    prompt = sample.input || sample[:input] || ""
    completion = " " <> (sample.output || sample[:output] || "")

    # Tokenize
    {:ok, prompt_tokens} = Tinkex.Tokenizer.encode(prompt, model_name, opts)
    {:ok, completion_tokens} = Tinkex.Tokenizer.encode(completion, model_name, opts)

    tokens = prompt_tokens ++ completion_tokens
    input_tokens = Enum.slice(tokens, 0..-2//1)
    target_tokens = Enum.slice(tokens, 1..-1//1)

    # Build weights (all 1.0 for basic case)
    # Could add citation weighting here later
    weights = List.duplicate(1.0, length(target_tokens))

    # Create Datum
    Tinkex.Types.Datum.new(%{
      model_input: Tinkex.Types.ModelInput.from_ints(input_tokens),
      loss_fn_inputs: %{
        target_tokens: target_tokens,
        weights: weights
      }
    })
  end
end
