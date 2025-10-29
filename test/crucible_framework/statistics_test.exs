defmodule CrucibleFramework.StatisticsTest do
  use ExUnit.Case
  doctest CrucibleFramework.Statistics

  alias CrucibleFramework.Statistics

  describe "mean/1" do
    test "calculates mean of integers" do
      assert Statistics.mean([1, 2, 3, 4, 5]) == 3.0
    end

    test "calculates mean of floats" do
      assert Statistics.mean([1.5, 2.5, 3.5]) == 2.5
    end

    test "returns nil for empty list" do
      assert Statistics.mean([]) == nil
    end

    test "handles single value" do
      assert Statistics.mean([42]) == 42.0
    end

    test "handles negative numbers" do
      assert Statistics.mean([-5, 0, 5]) == 0.0
    end
  end

  describe "median/1" do
    test "calculates median of odd-length list" do
      assert Statistics.median([1, 2, 3, 4, 5]) == 3
    end

    test "calculates median of even-length list" do
      assert Statistics.median([1, 2, 3, 4]) == 2.5
    end

    test "returns nil for empty list" do
      assert Statistics.median([]) == nil
    end

    test "handles single value" do
      assert Statistics.median([42]) == 42
    end

    test "handles unsorted data" do
      assert Statistics.median([5, 1, 3, 2, 4]) == 3
    end
  end

  describe "std_dev/1" do
    test "calculates standard deviation" do
      data = [2, 4, 4, 4, 5, 5, 7, 9]
      sd = Statistics.std_dev(data)
      assert_in_delta(sd, 2.14, 0.1)
    end

    test "returns nil for empty list" do
      assert Statistics.std_dev([]) == nil
    end

    test "returns 0 for single value" do
      assert Statistics.std_dev([42]) == 0.0
    end

    test "handles uniform data" do
      sd = Statistics.std_dev([5, 5, 5, 5, 5])
      assert_in_delta(sd, 0.0, 0.001)
    end
  end

  describe "variance/1" do
    test "calculates variance" do
      data = [2, 4, 4, 4, 5, 5, 7, 9]
      var = Statistics.variance(data)
      assert_in_delta(var, 4.57, 0.1)
    end

    test "returns nil for empty list" do
      assert Statistics.variance([]) == nil
    end

    test "returns 0 for single value" do
      assert Statistics.variance([42]) == 0.0
    end

    test "variance equals std_dev squared" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      var = Statistics.variance(data)
      sd = Statistics.std_dev(data)
      assert_in_delta(var, sd * sd, 0.001)
    end
  end

  describe "percentile/2" do
    test "calculates 50th percentile (median)" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      p50 = Statistics.percentile(data, 0.5)
      assert p50 == 5.5
    end

    test "calculates 95th percentile" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      p95 = Statistics.percentile(data, 0.95)
      assert_in_delta(p95, 9.55, 0.01)
    end

    test "calculates 99th percentile" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      p99 = Statistics.percentile(data, 0.99)
      assert p99 == 9.91
    end

    test "returns nil for empty list" do
      assert Statistics.percentile([], 0.5) == nil
    end

    test "handles 0th percentile" do
      data = [1, 2, 3, 4, 5]
      assert Statistics.percentile(data, 0.0) == 1
    end

    test "handles 100th percentile" do
      data = [1, 2, 3, 4, 5]
      assert Statistics.percentile(data, 1.0) == 5
    end
  end

  describe "summary/1" do
    test "calculates all summary statistics" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      summary = Statistics.summary(data)

      assert summary.count == 10
      assert summary.mean == 5.5
      assert summary.median == 5.5
      assert summary.min == 1
      assert summary.max == 10
      assert is_float(summary.std_dev)
      assert is_float(summary.variance)
      assert is_float(summary.p25)
      assert is_float(summary.p50)
      assert is_float(summary.p75)
      assert is_float(summary.p95)
      assert is_float(summary.p99)
    end

    test "returns nil for empty list" do
      assert Statistics.summary([]) == nil
    end

    test "handles single value" do
      summary = Statistics.summary([42])
      assert summary.count == 1
      assert summary.mean == 42.0
      assert summary.median == 42
      assert summary.min == 42
      assert summary.max == 42
      assert summary.std_dev == 0.0
    end

    test "percentiles are in correct order" do
      data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      summary = Statistics.summary(data)

      assert summary.p25 < summary.p50
      assert summary.p50 < summary.p75
      assert summary.p75 < summary.p95
      assert summary.p95 < summary.p99
    end
  end
end
