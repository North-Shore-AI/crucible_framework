defmodule Crucible.Datasets.BatchIterator do
  @moduledoc """
  Efficient batching with shuffling and prefetching.

  Provides an iterator interface for batched dataset access with support for:
  - Configurable batch sizes
  - Shuffling with reproducible seeds
  - Dropping incomplete final batches
  - Enumerable protocol for seamless integration
  """

  defstruct [
    :dataset,
    :batch_size,
    :shuffle,
    :seed,
    :drop_last,
    :prefetch,
    :current_index,
    :indices
  ]

  @type t :: %__MODULE__{
          dataset: list(),
          batch_size: pos_integer(),
          shuffle: boolean(),
          seed: integer() | nil,
          drop_last: boolean(),
          prefetch: pos_integer(),
          current_index: non_neg_integer(),
          indices: [non_neg_integer()]
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Creates a new batch iterator.

  ## Options
  - `:batch_size` - Number of examples per batch. Default: 32
  - `:shuffle` - Whether to shuffle the dataset. Default: false
  - `:seed` - Random seed for shuffling
  - `:drop_last` - Drop incomplete final batch. Default: false
  - `:prefetch` - Number of batches to prefetch. Default: 2

  ## Examples

      iterator = BatchIterator.new(dataset, batch_size: 16, shuffle: true)
      iterator = BatchIterator.new(dataset, batch_size: 32, drop_last: true)
  """
  @spec new(list(), keyword()) :: t()
  def new(dataset, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    shuffle = Keyword.get(opts, :shuffle, false)
    seed = Keyword.get(opts, :seed)
    drop_last = Keyword.get(opts, :drop_last, false)
    prefetch = Keyword.get(opts, :prefetch, 2)

    n = length(dataset)
    indices = Enum.to_list(0..(n - 1))

    indices =
      if shuffle do
        shuffle_indices(indices, seed)
      else
        indices
      end

    %__MODULE__{
      dataset: dataset,
      batch_size: batch_size,
      shuffle: shuffle,
      seed: seed,
      drop_last: drop_last,
      prefetch: prefetch,
      current_index: 0,
      indices: indices
    }
  end

  @doc """
  Returns the next batch and updated iterator.

  Returns `nil` when the iterator is exhausted.
  """
  @spec next(t()) :: {list(), t()} | nil
  def next(%__MODULE__{} = iterator) do
    batch_indices = get_batch_indices(iterator)

    if batch_indices == [] do
      nil
    else
      batch =
        Enum.map(batch_indices, fn idx ->
          Enum.at(iterator.dataset, idx)
        end)

      :telemetry.execute(
        [:crucible, :datasets, :batch],
        %{batch_size: length(batch)},
        %{index: iterator.current_index}
      )

      updated = %{iterator | current_index: iterator.current_index + iterator.batch_size}
      {batch, updated}
    end
  end

  @doc """
  Resets the iterator to the beginning.

  ## Options
  - `:reshuffle` - Reshuffle indices. Default: false
  - `:seed` - New seed for reshuffling
  """
  @spec reset(t(), keyword()) :: t()
  def reset(%__MODULE__{} = iterator, opts \\ []) do
    reshuffle = Keyword.get(opts, :reshuffle, false)
    new_seed = Keyword.get(opts, :seed, iterator.seed)

    indices =
      if reshuffle and iterator.shuffle do
        shuffle_indices(iterator.indices, new_seed)
      else
        iterator.indices
      end

    %{iterator | current_index: 0, indices: indices, seed: new_seed}
  end

  @doc """
  Returns true if all batches have been consumed.
  """
  @spec epoch_complete?(t()) :: boolean()
  def epoch_complete?(%__MODULE__{} = iterator) do
    remaining = length(iterator.indices) - iterator.current_index

    if iterator.drop_last do
      remaining < iterator.batch_size
    else
      remaining <= 0
    end
  end

  @doc """
  Returns the total number of batches in an epoch.
  """
  @spec num_batches(t()) :: non_neg_integer()
  def num_batches(%__MODULE__{} = iterator) do
    n = length(iterator.indices)

    if iterator.drop_last do
      div(n, iterator.batch_size)
    else
      ceil(n / iterator.batch_size)
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp shuffle_indices(indices, seed) do
    if seed do
      :rand.seed(:exsss, {seed, seed, seed})
    end

    Enum.shuffle(indices)
  end

  defp get_batch_indices(%__MODULE__{} = iterator) do
    start_idx = iterator.current_index
    end_idx = min(start_idx + iterator.batch_size, length(iterator.indices))

    if start_idx >= length(iterator.indices) do
      []
    else
      batch_indices = Enum.slice(iterator.indices, start_idx, end_idx - start_idx)

      # Check if we should drop this batch (incomplete and drop_last is true)
      if iterator.drop_last and length(batch_indices) < iterator.batch_size do
        []
      else
        batch_indices
      end
    end
  end
end

# Enumerable implementation
defimpl Enumerable, for: Crucible.Datasets.BatchIterator do
  alias Crucible.Datasets.BatchIterator

  def count(iterator) do
    {:ok, BatchIterator.num_batches(iterator)}
  end

  def member?(_iterator, _element) do
    {:error, __MODULE__}
  end

  def reduce(iterator, acc, fun) do
    reduce_batches(iterator, acc, fun)
  end

  def slice(_iterator) do
    {:error, __MODULE__}
  end

  defp reduce_batches(_iterator, {:halt, acc}, _fun) do
    {:halted, acc}
  end

  defp reduce_batches(iterator, {:suspend, acc}, fun) do
    {:suspended, acc, &reduce_batches(iterator, &1, fun)}
  end

  defp reduce_batches(iterator, {:cont, acc}, fun) do
    case BatchIterator.next(iterator) do
      nil ->
        {:done, acc}

      {batch, updated_iterator} ->
        reduce_batches(updated_iterator, fun.(batch, acc), fun)
    end
  end
end
