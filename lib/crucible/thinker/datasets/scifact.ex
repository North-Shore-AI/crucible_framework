defmodule Crucible.Thinker.Datasets.Scifact do
  @moduledoc """
  SciFact dataset loader for claim verification tasks.

  Provides sample data for testing and development.
  In production, would integrate with crucible_datasets for
  full dataset loading with caching and validation.
  """

  @doc """
  Loads SciFact dataset with optional limit.

  ## Options

  - `:split` - Dataset split (:train, :validation, :test)
  - `:limit` - Max samples to load
  - `:shuffle` - Random shuffle (default: false)

  ## Examples

      iex> {:ok, dataset} = Crucible.Thinker.Datasets.Scifact.load(limit: 5)
      iex> length(dataset) <= 5
      true

  """
  @spec load(keyword()) :: {:ok, [map()]} | {:error, term()}
  def load(opts \\ []) do
    limit = Keyword.get(opts, :limit)
    _split = Keyword.get(opts, :split, :train)

    samples = sample_data()

    result =
      if limit do
        Enum.take(samples, limit)
      else
        samples
      end

    {:ok, result}
  end

  @doc """
  Formats a sample for training with input and output (Tinkex format).

  ## Examples

      iex> sample = %{id: 1, claim: "Test", evidence: [%{doc_id: 1, text: "Evidence"}], cited_doc_ids: [1]}
      iex> formatted = Crucible.Thinker.Datasets.Scifact.format_for_training(sample)
      iex> Map.has_key?(formatted, :input)
      true

  """
  @spec format_for_training(map()) :: map()
  def format_for_training(sample) do
    %{
      input: build_prompt(sample),
      output: build_expected_output(sample)
    }
  end

  @doc """
  Returns sample dataset for testing and development.

  These are representative examples based on SciFact format.
  """
  @spec sample_data() :: [map()]
  def sample_data do
    [
      %{
        id: 1,
        claim:
          "Vitamin D deficiency is associated with increased risk of respiratory infections.",
        evidence: [
          %{
            doc_id: 12345,
            text:
              "Multiple studies have shown that low serum vitamin D levels correlate with higher incidence of respiratory tract infections."
          },
          %{
            doc_id: 12345,
            text:
              "Supplementation with vitamin D reduced the risk of acute respiratory infection by 12% in the meta-analysis."
          }
        ],
        cited_doc_ids: [12345],
        label: "SUPPORT"
      },
      %{
        id: 2,
        claim:
          "Exercise training improves cognitive function in patients with mild cognitive impairment.",
        evidence: [
          %{
            doc_id: 23456,
            text:
              "A 6-month aerobic exercise program resulted in significant improvements in memory and executive function in MCI patients."
          }
        ],
        cited_doc_ids: [23456],
        label: "SUPPORT"
      },
      %{
        id: 3,
        claim: "Gut microbiome composition affects response to cancer immunotherapy.",
        evidence: [
          %{
            doc_id: 34567,
            text:
              "Patients with higher diversity in gut microbiota showed better response to PD-1 blockade therapy."
          },
          %{
            doc_id: 34567,
            text:
              "Fecal microbiome transplantation from responders to non-responders improved treatment outcomes."
          }
        ],
        cited_doc_ids: [34567],
        label: "SUPPORT"
      },
      %{
        id: 4,
        claim: "Sleep deprivation impairs immune system function.",
        evidence: [
          %{
            doc_id: 45678,
            text:
              "Participants with less than 6 hours of sleep showed reduced T-cell proliferation and cytokine production."
          }
        ],
        cited_doc_ids: [45678],
        label: "SUPPORT"
      },
      %{
        id: 5,
        claim: "Mediterranean diet reduces cardiovascular disease risk.",
        evidence: [
          %{
            doc_id: 56789,
            text:
              "The PREDIMED trial demonstrated a 30% reduction in cardiovascular events with Mediterranean diet supplemented with olive oil or nuts."
          }
        ],
        cited_doc_ids: [56789],
        label: "SUPPORT"
      },
      %{
        id: 6,
        claim: "Antibiotic use in early childhood increases obesity risk.",
        evidence: [
          %{
            doc_id: 67890,
            text:
              "Children receiving antibiotics before age 2 had 11% higher risk of obesity by age 5 compared to unexposed children."
          }
        ],
        cited_doc_ids: [67890],
        label: "SUPPORT"
      },
      %{
        id: 7,
        claim: "Mindfulness meditation reduces symptoms of anxiety and depression.",
        evidence: [
          %{
            doc_id: 78901,
            text:
              "An 8-week mindfulness-based stress reduction program showed significant reductions in anxiety scores compared to control."
          }
        ],
        cited_doc_ids: [78901],
        label: "SUPPORT"
      },
      %{
        id: 8,
        claim: "Air pollution exposure is linked to neurodegenerative disease risk.",
        evidence: [
          %{
            doc_id: 89012,
            text:
              "Long-term exposure to fine particulate matter was associated with accelerated cognitive decline and increased dementia incidence."
          }
        ],
        cited_doc_ids: [89012],
        label: "SUPPORT"
      },
      %{
        id: 9,
        claim: "Intermittent fasting improves metabolic health markers.",
        evidence: [
          %{
            doc_id: 90123,
            text:
              "Time-restricted eating protocols resulted in improved insulin sensitivity and reduced inflammatory markers."
          }
        ],
        cited_doc_ids: [90123],
        label: "SUPPORT"
      },
      %{
        id: 10,
        claim: "Social isolation increases mortality risk in older adults.",
        evidence: [
          %{
            doc_id: 101_234,
            text:
              "Socially isolated older adults had 26% higher mortality risk over the 7-year follow-up period."
          }
        ],
        cited_doc_ids: [101_234],
        label: "SUPPORT"
      }
    ]
  end

  defp build_prompt(sample) do
    doc_ids = Enum.join(sample.cited_doc_ids, ", ")

    """
    Given the following scientific claim, extract structured claims with citations.

    Claim: #{sample.claim}

    Evidence documents: #{doc_ids}

    Extract claims in format: CLAIM[c*]: <claim text> (citing <doc_id>)
    """
  end

  defp build_expected_output(sample) do
    sample.evidence
    |> Enum.with_index(1)
    |> Enum.map(fn {ev, idx} ->
      "CLAIM[c#{idx}]: #{ev.text} (citing #{ev.doc_id})"
    end)
    |> Enum.join("\n")
  end
end
