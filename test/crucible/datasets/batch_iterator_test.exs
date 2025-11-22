defmodule Crucible.Datasets.BatchIteratorTest do
  use ExUnit.Case, async: true

  alias Crucible.Datasets.BatchIterator

  setup do
    dataset = Enum.map(1..100, fn i -> %{id: i, value: i * 2} end)
    {:ok, dataset: dataset}
  end

  describe "new/2" do
    test "creates iterator with batch size", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10)

      assert %BatchIterator{} = iterator
      assert iterator.batch_size == 10
      assert iterator.current_index == 0
    end

    test "uses default batch size when not specified", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset)

      assert iterator.batch_size == 32
    end

    test "shuffles when requested", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, shuffle: true, seed: 42)

      assert iterator.shuffle == true
      # Indices should be shuffled
      assert iterator.indices != Enum.to_list(0..(length(dataset) - 1))
    end

    test "preserves order when shuffle is false", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, shuffle: false)

      assert iterator.indices == Enum.to_list(0..(length(dataset) - 1))
    end

    test "respects drop_last option", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 30, drop_last: true)

      assert iterator.drop_last == true
    end
  end

  describe "next/1" do
    test "returns next batch", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10, shuffle: false)

      {batch, _iterator2} = BatchIterator.next(iterator)

      assert length(batch) == 10
      assert hd(batch).id == 1
    end

    test "advances iterator position", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10, shuffle: false)

      {_batch1, iterator2} = BatchIterator.next(iterator)
      {batch2, _iterator3} = BatchIterator.next(iterator2)

      assert hd(batch2).id == 11
    end

    test "returns nil when exhausted", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 50, shuffle: false)

      {_batch1, iterator2} = BatchIterator.next(iterator)
      {_batch2, iterator3} = BatchIterator.next(iterator2)
      result = BatchIterator.next(iterator3)

      assert result == nil
    end

    test "handles partial last batch when drop_last is false", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 30, drop_last: false)

      {batch1, it2} = BatchIterator.next(iterator)
      {batch2, it3} = BatchIterator.next(it2)
      {batch3, it4} = BatchIterator.next(it3)
      {batch4, _it5} = BatchIterator.next(it4)

      assert length(batch1) == 30
      assert length(batch2) == 30
      assert length(batch3) == 30
      assert length(batch4) == 10
    end

    test "drops partial last batch when drop_last is true", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 30, drop_last: true)

      {batch1, it2} = BatchIterator.next(iterator)
      {batch2, it3} = BatchIterator.next(it2)
      {batch3, it4} = BatchIterator.next(it3)
      result = BatchIterator.next(it4)

      assert length(batch1) == 30
      assert length(batch2) == 30
      assert length(batch3) == 30
      assert result == nil
    end
  end

  describe "reset/1" do
    test "resets iterator to beginning", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10, shuffle: false)

      {_batch, iterator2} = BatchIterator.next(iterator)
      iterator3 = BatchIterator.reset(iterator2)

      assert iterator3.current_index == 0
    end

    test "reshuffles indices when shuffle is enabled", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10, shuffle: true, seed: 42)
      old_indices = iterator.indices

      {_batch, iterator2} = BatchIterator.next(iterator)
      iterator3 = BatchIterator.reset(iterator2, reshuffle: true, seed: 123)

      assert iterator3.indices != old_indices
    end
  end

  describe "epoch_complete?/1" do
    test "returns false when batches remain", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 50)

      refute BatchIterator.epoch_complete?(iterator)
    end

    test "returns true when all batches consumed", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 50)

      {_batch1, iterator2} = BatchIterator.next(iterator)
      {_batch2, iterator3} = BatchIterator.next(iterator2)

      assert BatchIterator.epoch_complete?(iterator3)
    end
  end

  describe "num_batches/1" do
    test "calculates correct number of batches", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10)
      assert BatchIterator.num_batches(iterator) == 10
    end

    test "rounds up when drop_last is false", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 30, drop_last: false)
      assert BatchIterator.num_batches(iterator) == 4
    end

    test "rounds down when drop_last is true", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 30, drop_last: true)
      assert BatchIterator.num_batches(iterator) == 3
    end
  end

  describe "Enumerable" do
    test "iterates over batches", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 25)

      batches = Enum.to_list(iterator)

      assert length(batches) == 4
    end

    test "supports reduce", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 20)

      total =
        Enum.reduce(iterator, 0, fn batch, acc ->
          acc + length(batch)
        end)

      assert total == 100
    end

    test "supports count", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10)

      assert Enum.count(iterator) == 10
    end

    test "supports take", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10)

      batches = Enum.take(iterator, 3)

      assert length(batches) == 3
    end

    test "supports map", %{dataset: dataset} do
      iterator = BatchIterator.new(dataset, batch_size: 10)

      batch_sizes = Enum.map(iterator, &length/1)

      assert batch_sizes == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    end
  end
end
