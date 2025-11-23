# Semantic Validation Pipeline

## Overview

4-stage validation pipeline: Schema → Citation → Entailment → Similarity

Initial implementation uses Tinkex API; designed for Bumblebee swap.

## Pipeline Architecture

```elixir
defmodule Crucible.Thinker.Validation.Pipeline do
  @moduledoc """
  Orchestrates 4-stage semantic validation.
  """

  alias Crucible.Thinker.Validation.{Schema, Citation, Entailment, Similarity}

  @doc """
  Run full validation pipeline on model output.

  Returns aggregate scores and per-claim details.
  """
  def validate(output, context) do
    claims = Schema.parse(output)

    results = Enum.map(claims, fn claim ->
      %{
        claim: claim,
        schema_valid: Schema.validate(claim),
        citation_valid: Citation.verify(claim, context.corpus),
        entailment_score: Entailment.score(claim, context.evidence),
        similarity_score: Similarity.score(claim, context.expected)
      }
    end)

    emit_telemetry(results)

    %{
      claims: results,
      aggregate: aggregate_scores(results)
    }
  end

  defp aggregate_scores(results) do
    count = length(results)

    %{
      schema_compliance: Enum.count(results, & &1.schema_valid) / count,
      citation_accuracy: Enum.count(results, & &1.citation_valid) / count,
      mean_entailment: Enum.sum(Enum.map(results, & &1.entailment_score)) / count,
      mean_similarity: Enum.sum(Enum.map(results, & &1.similarity_score)) / count
    }
  end

  defp emit_telemetry(results) do
    :telemetry.execute(
      [:crucible, :thinker, :validation, :complete],
      aggregate_scores(results),
      %{claim_count: length(results)}
    )
  end
end
```

## Stage 1: Schema Validation

```elixir
defmodule Crucible.Thinker.Validation.Schema do
  @moduledoc """
  Validates CLAIM[c*] structure format.
  """

  @claim_pattern ~r/CLAIM\[c(\d+)\]:\s*(.+?)\s*\(citing\s+(\d+)\)/

  def parse(output) do
    @claim_pattern
    |> Regex.scan(output, capture: :all_but_first)
    |> Enum.map(fn [index, text, doc_id] ->
      %{
        index: String.to_integer(index),
        text: String.trim(text),
        doc_id: String.to_integer(doc_id)
      }
    end)
  end

  def validate(claim) do
    claim.index > 0 and
    String.length(claim.text) > 0 and
    claim.doc_id > 0
  end
end
```

## Stage 2: Citation Verification

```elixir
defmodule Crucible.Thinker.Validation.Citation do
  @moduledoc """
  Verifies citations exist in corpus.
  """

  def verify(claim, corpus) do
    case Map.get(corpus, claim.doc_id) do
      nil -> false
      doc -> text_in_document?(claim.text, doc)
    end
  end

  defp text_in_document?(claim_text, doc) do
    # Check if claim semantically relates to document
    # Simple: keyword overlap > threshold
    claim_words = tokenize(claim_text)
    doc_words = tokenize(doc.text)

    overlap = MapSet.intersection(
      MapSet.new(claim_words),
      MapSet.new(doc_words)
    )

    MapSet.size(overlap) / length(claim_words) > 0.3
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/)
    |> Enum.reject(&(&1 == ""))
  end
end
```

## Stage 3: Entailment Scoring

### Tinkex Implementation (Current)

```elixir
defmodule Crucible.Thinker.Validation.Entailment do
  @moduledoc """
  NLI-based entailment scoring.

  Uses Tinkex API initially, designed for Bumblebee swap.
  """

  @behaviour Crucible.Thinker.Validation.EntailmentBehaviour

  # Switch implementation via config
  @impl_module Application.compile_env(
    :crucible,
    :entailment_impl,
    Crucible.Thinker.Validation.Entailment.Tinkex
  )

  defdelegate score(claim, evidence), to: @impl_module
end

defmodule Crucible.Thinker.Validation.Entailment.Tinkex do
  @moduledoc """
  Tinkex-based entailment via DeBERTa-v3-large-mnli.
  """

  @base_url Application.compile_env(:crucible, :tinkex_url)

  def score(claim, evidence) do
    premise = evidence_to_premise(evidence)

    payload = %{
      "premise" => premise,
      "hypothesis" => claim.text,
      "model" => "microsoft/deberta-v3-large-mnli"
    }

    case Req.post("#{@base_url}/v1/predict/nli", json: payload) do
      {:ok, %{body: %{"entailment" => score}}} ->
        emit_telemetry(claim, score)
        score
      {:error, _} ->
        0.0
    end
  end

  defp evidence_to_premise(evidence) do
    evidence
    |> Enum.map(& &1.text)
    |> Enum.join(" ")
  end

  defp emit_telemetry(claim, score) do
    :telemetry.execute(
      [:crucible, :thinker, :validation, :entailment],
      %{score: score},
      %{claim_index: claim.index}
    )
  end
end
```

### Bumblebee Implementation (Future)

```elixir
defmodule Crucible.Thinker.Validation.Entailment.Bumblebee do
  @moduledoc """
  Local Bumblebee-based entailment.
  """

  def score(claim, evidence) do
    serving = get_or_start_serving()
    premise = evidence_to_premise(evidence)

    input = %{
      "premise" => premise,
      "hypothesis" => claim.text
    }

    %{predictions: [%{label: label, score: score}]} =
      Nx.Serving.batched_run(serving, input)

    if label == "entailment", do: score, else: 1.0 - score
  end

  defp get_or_start_serving do
    case Process.whereis(:entailment_serving) do
      nil -> start_serving()
      pid -> GenServer.call(pid, :get_serving)
    end
  end

  defp start_serving do
    {:ok, model} = Bumblebee.load_model(
      {:hf, "microsoft/deberta-v3-large-mnli"}
    )
    {:ok, tokenizer} = Bumblebee.load_tokenizer(
      {:hf, "microsoft/deberta-v3-large-mnli"}
    )

    Bumblebee.Text.text_classification(model, tokenizer,
      compile: [batch_size: 8],
      defn_options: [compiler: EXLA]
    )
  end

  defp evidence_to_premise(evidence) do
    evidence |> Enum.map(& &1.text) |> Enum.join(" ")
  end
end
```

## Stage 4: Similarity Scoring

### Tinkex Implementation (Current)

```elixir
defmodule Crucible.Thinker.Validation.Similarity do
  @moduledoc """
  Embedding-based similarity scoring.
  """

  @impl_module Application.compile_env(
    :crucible,
    :similarity_impl,
    Crucible.Thinker.Validation.Similarity.Tinkex
  )

  defdelegate score(claim, expected), to: @impl_module
end

defmodule Crucible.Thinker.Validation.Similarity.Tinkex do
  @base_url Application.compile_env(:crucible, :tinkex_url)

  def score(claim, expected) do
    with {:ok, claim_emb} <- embed(claim.text),
         {:ok, expected_emb} <- embed(expected) do
      cosine_similarity(claim_emb, expected_emb)
    else
      _ -> 0.0
    end
  end

  defp embed(text) do
    payload = %{
      "text" => text,
      "model" => "sentence-transformers/all-MiniLM-L6-v2"
    }

    case Req.post("#{@base_url}/v1/predict/embed", json: payload) do
      {:ok, %{body: %{"embedding" => emb}}} -> {:ok, emb}
      {:error, reason} -> {:error, reason}
    end
  end

  defp cosine_similarity(a, b) do
    dot = Enum.zip(a, b) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
    norm_a = :math.sqrt(Enum.map(a, &(&1 * &1)) |> Enum.sum())
    norm_b = :math.sqrt(Enum.map(b, &(&1 * &1)) |> Enum.sum())
    dot / (norm_a * norm_b)
  end
end
```

### Bumblebee Implementation (Future)

```elixir
defmodule Crucible.Thinker.Validation.Similarity.Bumblebee do
  def score(claim, expected) do
    serving = get_or_start_serving()

    %{embedding: claim_emb} = Nx.Serving.batched_run(serving, claim.text)
    %{embedding: expected_emb} = Nx.Serving.batched_run(serving, expected)

    Nx.dot(claim_emb, expected_emb)
    |> Nx.divide(Nx.multiply(Nx.LinAlg.norm(claim_emb), Nx.LinAlg.norm(expected_emb)))
    |> Nx.to_number()
  end

  defp get_or_start_serving do
    # Similar to Entailment.Bumblebee
    {:ok, model} = Bumblebee.load_model(
      {:hf, "sentence-transformers/all-MiniLM-L6-v2"}
    )
    {:ok, tokenizer} = Bumblebee.load_tokenizer(
      {:hf, "sentence-transformers/all-MiniLM-L6-v2"}
    )

    Bumblebee.Text.TextEmbedding.text_embedding(model, tokenizer,
      compile: [batch_size: 8],
      defn_options: [compiler: EXLA]
    )
  end
end
```

## Configuration

```elixir
# config/config.exs

config :crucible,
  tinkex_url: "http://localhost:8080",

  # Switch to Bumblebee implementations when ready
  entailment_impl: Crucible.Thinker.Validation.Entailment.Tinkex,
  similarity_impl: Crucible.Thinker.Validation.Similarity.Tinkex
```

## Usage

```elixir
alias Crucible.Thinker.Validation.Pipeline

output = """
CLAIM[c1]: The study found significant results (citing 12345)
CLAIM[c2]: Patient outcomes improved by 20% (citing 12346)
"""

context = %{
  corpus: corpus_map,
  evidence: evidence_list,
  expected: "Expected claim text..."
}

%{aggregate: scores, claims: details} = Pipeline.validate(output, context)

# scores = %{
#   schema_compliance: 1.0,
#   citation_accuracy: 1.0,
#   mean_entailment: 0.72,
#   mean_similarity: 0.85
# }
```
