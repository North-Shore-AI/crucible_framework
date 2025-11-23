defmodule Crucible.Thinker.Validation.Entailment.Bumblebee do
  @moduledoc """
  Bumblebee-based local entailment scoring using DeBERTa-v3-large-mnli.

  Requires Bumblebee and EXLA dependencies.
  """

  @behaviour Crucible.Thinker.Validation.EntailmentBehaviour

  require Logger

  @default_model {:hf, "microsoft/deberta-v3-large-mnli"}

  @impl true
  def score(claim, evidence) when is_list(evidence) do
    if Enum.empty?(evidence) do
      0.0
    else
      premise = evidence_to_premise(evidence)

      case run_inference(premise, claim.text) do
        {:ok, result} ->
          score = extract_entailment_score(result)
          emit_telemetry(claim, score)
          score

        {:error, reason} ->
          Logger.warning("Bumblebee inference error: #{inspect(reason)}")
          0.0
      end
    end
  end

  @impl true
  def classify(claim, evidence) do
    if Enum.empty?(evidence) do
      %{label: :neutral, score: 0.0}
    else
      premise = evidence_to_premise(evidence)

      case run_inference(premise, claim.text) do
        {:ok, result} ->
          extract_classification(result)

        {:error, _reason} ->
          %{label: :neutral, score: 0.0}
      end
    end
  end

  defp run_inference(premise, hypothesis) do
    serving = get_or_start_serving()

    if serving do
      input = %{
        sequence: "#{premise} [SEP] #{hypothesis}",
        # For sequence classification models
        second_sequence: nil
      }

      try do
        result = Nx.Serving.batched_run(serving, input)
        {:ok, result}
      rescue
        e -> {:error, e}
      end
    else
      {:error, :serving_not_available}
    end
  end

  defp get_or_start_serving do
    case Process.whereis(:thinker_entailment_serving) do
      nil ->
        start_serving()

      pid ->
        if Process.alive?(pid) do
          GenServer.call(pid, :get_serving)
        else
          start_serving()
        end
    end
  end

  defp start_serving do
    if Code.ensure_loaded?(Bumblebee) do
      try do
        {:ok, model_info} = Bumblebee.load_model(@default_model)
        {:ok, tokenizer} = Bumblebee.load_tokenizer(@default_model)

        serving =
          Bumblebee.Text.text_classification(model_info, tokenizer,
            compile: [batch_size: 8],
            defn_options: [compiler: EXLA]
          )

        # Start as named process
        {:ok, _pid} =
          Nx.Serving.start_link(
            serving: serving,
            name: :thinker_entailment_serving,
            batch_size: 8,
            batch_timeout: 100
          )

        serving
      rescue
        e ->
          Logger.error("Failed to start Bumblebee serving: #{inspect(e)}")
          nil
      end
    else
      Logger.warning("Bumblebee not available")
      nil
    end
  end

  defp extract_entailment_score(%{predictions: predictions}) do
    predictions
    |> Enum.find(fn p -> p.label == "entailment" or p.label == "ENTAILMENT" end)
    |> case do
      nil -> 0.0
      pred -> pred.score
    end
  end

  defp extract_entailment_score(_), do: 0.0

  defp extract_classification(%{predictions: predictions}) do
    top = Enum.max_by(predictions, & &1.score)

    label =
      top.label
      |> String.downcase()
      |> String.to_atom()

    %{label: label, score: top.score}
  end

  defp extract_classification(_), do: %{label: :neutral, score: 0.0}

  defp evidence_to_premise(evidence) do
    evidence
    |> Enum.map(& &1.text)
    |> Enum.join(" ")
  end

  defp emit_telemetry(claim, score) do
    :telemetry.execute(
      [:crucible, :thinker, :validation, :entailment, :bumblebee],
      %{score: score},
      %{claim_index: claim.index}
    )
  end
end
