defmodule CrucibleFramework.Statistics do
  @moduledoc """
  Basic statistical functions for research analysis.

  This module provides fundamental statistical calculations used in
  LLM reliability research. For advanced statistical testing and
  publication-quality analysis, use the Bench library.

  ## Examples

      iex> data = [1, 2, 3, 4, 5]
      iex> CrucibleFramework.Statistics.mean(data)
      3.0
  """

  @doc """
  Calculates the arithmetic mean of a list of numbers.

  ## Examples

      iex> CrucibleFramework.Statistics.mean([1, 2, 3, 4, 5])
      3.0

      iex> CrucibleFramework.Statistics.mean([10, 20, 30])
      20.0

      iex> CrucibleFramework.Statistics.mean([])
      nil
  """
  @spec mean([number()]) :: float() | nil
  def mean([]), do: nil

  def mean(values) when is_list(values) do
    sum = Enum.sum(values)
    count = length(values)
    sum / count
  end

  @doc """
  Calculates the median of a list of numbers.

  ## Examples

      iex> CrucibleFramework.Statistics.median([1, 2, 3, 4, 5])
      3

      iex> CrucibleFramework.Statistics.median([1, 2, 3, 4])
      2.5

      iex> CrucibleFramework.Statistics.median([])
      nil
  """
  @spec median([number()]) :: number() | nil
  def median([]), do: nil

  def median(values) when is_list(values) do
    sorted = Enum.sort(values)
    count = length(sorted)
    middle = div(count, 2)

    if rem(count, 2) == 0 do
      # Even number of elements - average the two middle values
      (Enum.at(sorted, middle - 1) + Enum.at(sorted, middle)) / 2
    else
      # Odd number of elements - take the middle value
      Enum.at(sorted, middle)
    end
  end

  @doc """
  Calculates the standard deviation of a list of numbers.

  ## Examples

      iex> data = [2, 4, 4, 4, 5, 5, 7, 9]
      iex> sd = CrucibleFramework.Statistics.std_dev(data)
      iex> Float.round(sd, 2)
      2.14

      iex> CrucibleFramework.Statistics.std_dev([])
      nil
  """
  @spec std_dev([number()]) :: float() | nil
  def std_dev([]), do: nil
  def std_dev([_]), do: 0.0

  def std_dev(values) when is_list(values) do
    avg = mean(values)
    variance = variance(values, avg)
    :math.sqrt(variance)
  end

  @doc """
  Calculates the variance of a list of numbers.

  ## Examples

      iex> data = [2, 4, 4, 4, 5, 5, 7, 9]
      iex> var = CrucibleFramework.Statistics.variance(data)
      iex> Float.round(var, 2)
      4.57

      iex> CrucibleFramework.Statistics.variance([])
      nil
  """
  @spec variance([number()]) :: float() | nil
  def variance([]), do: nil
  def variance([_]), do: 0.0

  def variance(values) when is_list(values) do
    avg = mean(values)
    variance(values, avg)
  end

  defp variance(values, mean) do
    sum_squared_diff =
      values
      |> Enum.map(fn x -> :math.pow(x - mean, 2) end)
      |> Enum.sum()

    sum_squared_diff / (length(values) - 1)
  end

  @doc """
  Calculates a percentile of a list of numbers.

  ## Examples

      iex> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      iex> CrucibleFramework.Statistics.percentile(data, 0.5)
      5.5

      iex> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      iex> p95 = CrucibleFramework.Statistics.percentile(data, 0.95)
      iex> Float.round(p95, 2)
      9.55

      iex> CrucibleFramework.Statistics.percentile([], 0.5)
      nil
  """
  @spec percentile([number()], float()) :: float() | nil
  def percentile([], _p), do: nil

  def percentile(values, p) when is_list(values) and p >= 0 and p <= 1 do
    sorted = Enum.sort(values)
    count = length(sorted)
    index = p * (count - 1)
    lower_index = floor(index)
    upper_index = ceil(index)

    if lower_index == upper_index do
      Enum.at(sorted, round(index))
    else
      lower_value = Enum.at(sorted, lower_index)
      upper_value = Enum.at(sorted, upper_index)
      fraction = index - lower_index
      lower_value + fraction * (upper_value - lower_value)
    end
  end

  @doc """
  Calculates summary statistics for a list of numbers.

  Returns a map with mean, median, std_dev, min, max, and percentiles.

  ## Examples

      iex> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      iex> summary = CrucibleFramework.Statistics.summary(data)
      iex> summary.mean
      5.5
      iex> summary.min
      1
      iex> summary.max
      10
  """
  @spec summary([number()]) :: map() | nil
  def summary([]), do: nil

  def summary(values) when is_list(values) do
    %{
      count: length(values),
      mean: mean(values),
      median: median(values),
      std_dev: std_dev(values),
      variance: variance(values),
      min: Enum.min(values),
      max: Enum.max(values),
      p25: percentile(values, 0.25),
      p50: percentile(values, 0.50),
      p75: percentile(values, 0.75),
      p95: percentile(values, 0.95),
      p99: percentile(values, 0.99)
    }
  end
end
