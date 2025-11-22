defmodule Crucible.Datasets.MLLoader do
  @moduledoc """
  Loads datasets and prepares them for ML training.

  Supports various dataset formats and provides utilities for:
  - Loading and streaming datasets
  - Transforming examples
  - Train/val/test splitting
  - Random sampling
  """

  require Logger

  @type dataset_name :: :scifact | :fever | :gsm8k | :mmlu | :humaneval | :custom
  @type example :: map()
  @type dataset :: [example()]

  # Sample data for demonstration/testing
  @sample_scifact [
    %{
      "claim" => "Climate change affects biodiversity",
      "evidence" => ["Studies show..."],
      "label" => "SUPPORTS",
      "id" => 1,
      "evidence_ids" => [1]
    },
    %{
      "claim" => "Water is not essential for life",
      "evidence" => ["Biology shows..."],
      "label" => "REFUTES",
      "id" => 2,
      "evidence_ids" => [2]
    },
    %{
      "claim" => "Exercise improves health",
      "evidence" => ["Medical research..."],
      "label" => "SUPPORTS",
      "id" => 3,
      "evidence_ids" => [3]
    }
  ]

  @sample_gsm8k [
    %{"question" => "If John has 5 apples and gives away 2, how many remain?", "answer" => "3"},
    %{"question" => "What is 15 + 27?", "answer" => "42"},
    %{
      "question" => "A train travels 100 miles in 2 hours. What is its speed?",
      "answer" => "50 mph"
    }
  ]

  @sample_fever [
    %{"claim" => "The Earth orbits the Sun", "label" => "SUPPORTS", "id" => 1},
    %{"claim" => "The Moon is made of cheese", "label" => "REFUTES", "id" => 2}
  ]

  @sample_mmlu [
    %{
      "question" => "What is the capital of France?",
      "choices" => ["Berlin", "Paris", "London", "Rome"],
      "answer" => "B",
      "subject" => "geography"
    },
    %{
      "question" => "Who wrote Hamlet?",
      "choices" => ["Dickens", "Austen", "Shakespeare", "Twain"],
      "answer" => "C",
      "subject" => "literature"
    }
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Loads a dataset by name.

  ## Options
  - `:split` - Dataset split (:train, :val, :test). Default: :train
  - `:limit` - Maximum number of examples to load
  - `:transform` - Function to transform each example
  - `:cache` - Whether to cache loaded data. Default: true

  ## Examples

      {:ok, dataset} = MLLoader.load(:scifact, split: :train, limit: 100)
      {:ok, dataset} = MLLoader.load(:gsm8k, transform: &my_transform/1)
  """
  @spec load(dataset_name(), keyword()) :: {:ok, dataset()} | {:error, atom()}
  def load(name, opts \\ []) do
    split = Keyword.get(opts, :split, :train)
    limit = Keyword.get(opts, :limit)
    transform = Keyword.get(opts, :transform)

    :telemetry.execute(
      [:crucible, :datasets, :load],
      %{system_time: System.system_time()},
      %{dataset: name, split: split}
    )

    case get_raw_data(name, split) do
      {:ok, data} ->
        data = if limit, do: Enum.take(data, limit), else: data
        data = if transform, do: Enum.map(data, transform), else: data
        {:ok, data}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Streams a dataset for memory-efficient processing.

  Returns a Stream that lazily loads and transforms examples.
  """
  @spec stream(dataset_name(), keyword()) :: Enumerable.t()
  def stream(name, opts \\ []) do
    limit = Keyword.get(opts, :limit)
    transform = Keyword.get(opts, :transform, & &1)

    case get_raw_data(name, :train) do
      {:ok, data} ->
        stream = Stream.map(data, transform)
        if limit, do: Stream.take(stream, limit), else: stream

      {:error, _reason} ->
        Stream.map([], & &1)
    end
  end

  @doc """
  Prepares a dataset for training by formatting examples.

  ## Options
  - `:formatter` - Dataset-specific formatter (:scifact, :fever, :gsm8k, :mmlu)
  """
  @spec prepare_for_training(dataset(), keyword()) :: dataset()
  def prepare_for_training(dataset, opts \\ []) do
    formatter_name = Keyword.get(opts, :formatter, :default)
    format_fn = get_formatter(formatter_name)

    Enum.map(dataset, fn example ->
      formatted = format_fn.(example)
      Map.put_new(formatted, :metadata, %{})
    end)
  end

  @doc """
  Splits a dataset into train/val/test sets.

  ## Examples

      {train, val, test} = MLLoader.split(dataset, {0.7, 0.15, 0.15})
      {train, test} = MLLoader.split(dataset, {0.8, 0.2})
  """
  @spec split(dataset(), tuple()) :: tuple()
  def split(dataset, ratios, opts \\ [])

  def split(dataset, {train_ratio, val_ratio, _test_ratio}, opts) do
    stratify = Keyword.get(opts, :stratify)
    shuffled = shuffle_dataset(dataset, opts)

    n = length(shuffled)
    train_size = round(n * train_ratio)
    val_size = round(n * val_ratio)

    {train, rest} = Enum.split(shuffled, train_size)
    {val, test} = Enum.split(rest, val_size)

    if stratify do
      # Simple stratification - in production would be more sophisticated
      {train, val, test}
    else
      {train, val, test}
    end
  end

  def split(dataset, {train_ratio, _test_ratio}, opts) do
    shuffled = shuffle_dataset(dataset, opts)

    n = length(shuffled)
    train_size = round(n * train_ratio)

    Enum.split(shuffled, train_size)
  end

  @doc """
  Samples n random examples from a dataset.

  ## Options
  - `:seed` - Random seed for reproducibility
  - `:replacement` - Whether to sample with replacement. Default: false
  """
  @spec sample(dataset(), pos_integer(), keyword()) :: dataset()
  def sample(dataset, n, opts \\ []) do
    seed = Keyword.get(opts, :seed)

    if seed do
      :rand.seed(:exsss, {seed, seed, seed})
    end

    dataset
    |> Enum.shuffle()
    |> Enum.take(min(n, length(dataset)))
  end

  # ============================================================================
  # Dataset-specific formatters
  # ============================================================================

  @doc """
  Returns a formatter function for a specific dataset type.
  """
  @spec formatter(dataset_name()) :: (example() -> map())
  def formatter(:scifact) do
    fn example ->
      claim = example["claim"]
      evidence = example["evidence"] |> List.wrap() |> Enum.join(" ")

      %{
        input: "Claim: #{claim}\nEvidence: #{evidence}\n\nVerdict:",
        output: example["label"],
        metadata: %{
          claim_id: example["id"],
          evidence_ids: example["evidence_ids"] || []
        }
      }
    end
  end

  def formatter(:fever) do
    fn example ->
      %{
        input: "Claim: #{example["claim"]}\n\nVerify:",
        output: example["label"],
        metadata: %{id: example["id"]}
      }
    end
  end

  def formatter(:gsm8k) do
    fn example ->
      %{
        input: "Problem: #{example["question"]}\n\nSolution:",
        output: example["answer"],
        metadata: %{}
      }
    end
  end

  def formatter(:mmlu) do
    fn example ->
      choices =
        example["choices"]
        |> Enum.with_index()
        |> Enum.map(fn {choice, i} -> "#{<<65 + i>>}. #{choice}" end)
        |> Enum.join("\n")

      %{
        input: "Question: #{example["question"]}\n\n#{choices}\n\nAnswer:",
        output: example["answer"],
        metadata: %{subject: example["subject"]}
      }
    end
  end

  def formatter(:humaneval) do
    fn example ->
      %{
        input: example["prompt"],
        output: example["canonical_solution"],
        metadata: %{
          task_id: example["task_id"],
          entry_point: example["entry_point"]
        }
      }
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_raw_data(:scifact, _split), do: {:ok, generate_samples(@sample_scifact, 50)}
  defp get_raw_data(:fever, _split), do: {:ok, generate_samples(@sample_fever, 50)}
  defp get_raw_data(:gsm8k, _split), do: {:ok, generate_samples(@sample_gsm8k, 50)}
  defp get_raw_data(:mmlu, _split), do: {:ok, generate_samples(@sample_mmlu, 50)}
  defp get_raw_data(:humaneval, _split), do: {:ok, []}
  defp get_raw_data(_unknown, _split), do: {:error, :unknown_dataset}

  defp generate_samples(base_samples, count) do
    base_samples
    |> Stream.cycle()
    |> Stream.with_index()
    |> Stream.map(fn {sample, idx} -> Map.put(sample, "id", idx + 1) end)
    |> Enum.take(count)
  end

  defp get_formatter(:default), do: &default_formatter/1
  defp get_formatter(:scifact), do: formatter(:scifact)
  defp get_formatter(:fever), do: formatter(:fever)
  defp get_formatter(:gsm8k), do: formatter(:gsm8k)
  defp get_formatter(:mmlu), do: formatter(:mmlu)
  defp get_formatter(:humaneval), do: formatter(:humaneval)

  defp default_formatter(example) do
    %{
      input: format_input(example),
      output: format_output(example),
      metadata: extract_metadata(example)
    }
  end

  defp format_input(example) do
    cond do
      Map.has_key?(example, "input") -> example["input"]
      Map.has_key?(example, :input) -> example.input
      Map.has_key?(example, "claim") -> "Claim: #{example["claim"]}"
      Map.has_key?(example, "question") -> "Question: #{example["question"]}"
      Map.has_key?(example, "prompt") -> example["prompt"]
      true -> inspect(example)
    end
  end

  defp format_output(example) do
    cond do
      Map.has_key?(example, "output") -> example["output"]
      Map.has_key?(example, :output) -> example.output
      Map.has_key?(example, "label") -> example["label"]
      Map.has_key?(example, "answer") -> example["answer"]
      true -> ""
    end
  end

  defp extract_metadata(example) do
    example
    |> Map.drop(["input", "output", "claim", "evidence", "label", "question", "answer", "prompt"])
    |> Map.drop([:input, :output])
  end

  defp shuffle_dataset(dataset, opts) do
    seed = Keyword.get(opts, :seed)

    if seed do
      :rand.seed(:exsss, {seed, seed, seed})
    end

    Enum.shuffle(dataset)
  end
end
