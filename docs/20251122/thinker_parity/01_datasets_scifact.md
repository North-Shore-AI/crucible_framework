# SciFact Dataset Integration

## Overview

Load SciFact dataset through crucible_datasets with custom adapter.

## Module Design

```elixir
defmodule Crucible.Thinker.Datasets.SciFact do
  @moduledoc """
  SciFact dataset loader for claim verification tasks.
  Uses crucible_datasets for caching and version tracking.
  """

  alias Crucible.Datasets.DatasetManager

  @dataset_config %{
    name: :scifact,
    source: "allenai/scifact",
    version: "1.0.0",
    splits: [:train, :validation, :test],
    features: [:id, :claim, :evidence, :cited_doc_ids, :label]
  }

  @doc """
  Load SciFact dataset with optional limit.

  ## Options
  - `:split` - Dataset split (:train, :validation, :test)
  - `:limit` - Max samples to load
  - `:shuffle` - Random shuffle (default: false)

  ## Example
      {:ok, dataset} = SciFact.load(split: :train, limit: 15)
  """
  def load(opts \\ []) do
    split = Keyword.get(opts, :split, :train)
    limit = Keyword.get(opts, :limit, nil)

    with {:ok, raw_data} <- DatasetManager.load(@dataset_config, split),
         {:ok, validated} <- validate_with_datacheck(raw_data) do
      samples = maybe_limit(validated, limit)
      {:ok, samples}
    end
  end

  defp validate_with_datacheck(data) do
    alias ExDataCheck.Validator

    expectations = [
      {:expect_column_to_exist, :claim},
      {:expect_column_to_exist, :evidence},
      {:expect_column_values_to_not_be_null, :claim},
      {:expect_column_values_to_be_of_type, :id, :integer}
    ]

    case Validator.validate(data, expectations) do
      %{success: true} = result -> {:ok, data}
      %{success: false} = result -> {:error, {:validation_failed, result}}
    end
  end

  defp maybe_limit(data, nil), do: data
  defp maybe_limit(data, limit), do: Enum.take(data, limit)

  @doc """
  Format sample for training prompt.
  """
  def format_for_training(sample) do
    %{
      prompt: build_prompt(sample),
      expected_output: build_expected_output(sample)
    }
  end

  defp build_prompt(sample) do
    """
    Given the following scientific claim, extract structured claims with citations.

    Claim: #{sample.claim}

    Evidence documents: #{Enum.join(sample.cited_doc_ids, ", ")}

    Extract claims in format: CLAIM[c*]: <claim text> (citing <doc_id>)
    """
  end

  defp build_expected_output(sample) do
    # Build from evidence structure
    sample.evidence
    |> Enum.with_index(1)
    |> Enum.map(fn {ev, idx} ->
      "CLAIM[c#{idx}]: #{ev.text} (citing #{ev.doc_id})"
    end)
    |> Enum.join("\n")
  end
end
```

## crucible_datasets Adapter

Requires implementing adapter for SciFact format:

```elixir
defmodule Crucible.Datasets.Adapters.SciFact do
  @behaviour Crucible.Datasets.Adapter

  @impl true
  def fetch(config) do
    # Download from HuggingFace datasets
    url = "https://huggingface.co/datasets/allenai/scifact/resolve/main"

    {:ok, %{
      train: download_and_parse("#{url}/claims_train.jsonl"),
      validation: download_and_parse("#{url}/claims_dev.jsonl"),
      test: download_and_parse("#{url}/claims_test.jsonl"),
      corpus: download_and_parse("#{url}/corpus.jsonl")
    }}
  end

  @impl true
  def parse_sample(raw) do
    %{
      id: raw["id"],
      claim: raw["claim"],
      evidence: parse_evidence(raw["evidence"]),
      cited_doc_ids: raw["cited_doc_ids"] || [],
      label: raw["label"]
    }
  end

  defp parse_evidence(nil), do: []
  defp parse_evidence(evidence) do
    Enum.flat_map(evidence, fn {doc_id, sentences} ->
      Enum.map(sentences, fn sent ->
        %{doc_id: doc_id, sentence_id: sent["sentence_id"], text: sent["text"]}
      end)
    end)
  end
end
```

## Telemetry Events

```elixir
:telemetry.execute(
  [:crucible, :datasets, :scifact, :load],
  %{count: length(samples), duration: duration},
  %{split: split, limit: limit}
)
```

## Integration with crucible_harness

```elixir
Crucible.Harness.experiment "scifact-claim-extraction" do
  dataset do
    source Crucible.Thinker.Datasets.SciFact
    split :train
    limit 15
  end

  # ... rest of experiment
end
```
