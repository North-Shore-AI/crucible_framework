defmodule Crucible.Thinker.Validation.Similarity.Bumblebee do
  @moduledoc """
  Bumblebee-based local similarity scoring using sentence embeddings.

  Requires Bumblebee and EXLA dependencies.
  """

  @behaviour Crucible.Thinker.Validation.SimilarityBehaviour

  require Logger

  @default_model {:hf, "sentence-transformers/all-MiniLM-L6-v2"}

  @impl true
  def score(claim, expected) when is_binary(expected) do
    if expected == "" do
      0.0
    else
      case get_embeddings([claim.text, expected]) do
        {:ok, [claim_emb, expected_emb]} ->
          similarity = cosine_similarity_nx(claim_emb, expected_emb)
          emit_telemetry(claim, similarity)
          similarity

        {:error, reason} ->
          Logger.warning("Bumblebee embedding error: #{inspect(reason)}")
          0.0
      end
    end
  end

  @impl true
  def batch_score(claims, expected) when is_list(claims) and is_list(expected) do
    texts = Enum.map(claims, & &1.text) ++ expected

    case get_embeddings(texts) do
      {:ok, embeddings} ->
        n = length(claims)
        claim_embs = Enum.take(embeddings, n)
        expected_embs = Enum.drop(embeddings, n)

        claim_embs
        |> Enum.zip(expected_embs)
        |> Enum.zip(claims)
        |> Enum.map(fn {{c_emb, e_emb}, claim} ->
          score = cosine_similarity_nx(c_emb, e_emb)
          emit_telemetry(claim, score)
          score
        end)

      {:error, _reason} ->
        Enum.map(claims, fn _ -> 0.0 end)
    end
  end

  defp get_embeddings(texts) do
    serving = get_or_start_serving()

    if serving do
      try do
        results =
          Enum.map(texts, fn text ->
            %{embedding: emb} = Nx.Serving.batched_run(serving, text)
            emb
          end)

        {:ok, results}
      rescue
        e -> {:error, e}
      end
    else
      {:error, :serving_not_available}
    end
  end

  defp get_or_start_serving do
    case Process.whereis(:thinker_similarity_serving) do
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
          Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
            compile: [batch_size: 8],
            defn_options: [compiler: EXLA]
          )

        {:ok, _pid} =
          Nx.Serving.start_link(
            serving: serving,
            name: :thinker_similarity_serving,
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

  defp cosine_similarity_nx(a, b) do
    if Code.ensure_loaded?(Nx) do
      dot = Nx.dot(a, b) |> Nx.to_number()
      norm_a = Nx.LinAlg.norm(a) |> Nx.to_number()
      norm_b = Nx.LinAlg.norm(b) |> Nx.to_number()

      if norm_a == 0 or norm_b == 0 do
        0.0
      else
        dot / (norm_a * norm_b)
      end
    else
      0.0
    end
  end

  defp emit_telemetry(claim, score) do
    :telemetry.execute(
      [:crucible, :thinker, :validation, :similarity, :bumblebee],
      %{score: score},
      %{claim_index: claim.index}
    )
  end
end
