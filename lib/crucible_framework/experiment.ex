defmodule CrucibleFramework.Experiment do
  @moduledoc """
  A simple experiment definition module for demonstration purposes.

  This module provides a basic structure for defining and running experiments
  in the Crucible Framework. For full-featured experiment orchestration,
  use the ResearchHarness library.

  ## Example

      defmodule MyExperiment do
        use CrucibleFramework.Experiment

        def run do
          %{
            name: "My First Experiment",
            conditions: ["baseline", "treatment"],
            metrics: [:accuracy, :latency],
            status: :configured
          }
        end
      end
  """

  @doc """
  Defines an experiment configuration.
  """
  defmacro __using__(_opts) do
    quote do
      @behaviour CrucibleFramework.Experiment

      @doc """
      Returns the experiment configuration.
      """
      def config do
        %{
          name: name(),
          description: description(),
          conditions: conditions(),
          metrics: metrics(),
          status: :configured
        }
      end

      @doc """
      Returns the experiment name.
      Override this in your experiment module.
      """
      def name, do: "Unnamed Experiment"

      @doc """
      Returns the experiment description.
      Override this in your experiment module.
      """
      def description, do: "No description provided"

      @doc """
      Returns the list of experimental conditions.
      Override this in your experiment module.
      """
      def conditions, do: []

      @doc """
      Returns the list of metrics to collect.
      Override this in your experiment module.
      """
      def metrics, do: []

      defoverridable name: 0, description: 0, conditions: 0, metrics: 0
    end
  end

  @doc """
  Callback to define the experiment's run logic.
  """
  @callback run() :: {:ok, map()} | {:error, term()}
  @optional_callbacks run: 0

  @doc """
  Validates an experiment configuration.

  ## Examples

      iex> config = %{name: "Test", conditions: ["a", "b"], metrics: [:accuracy]}
      iex> CrucibleFramework.Experiment.validate(config)
      {:ok, config}

      iex> CrucibleFramework.Experiment.validate(%{})
      {:error, "Missing required field: name"}
  """
  @spec validate(map()) :: {:ok, map()} | {:error, String.t()}
  def validate(config) when is_map(config) do
    required_fields = [:name, :conditions, :metrics]

    case find_missing_field(config, required_fields) do
      nil -> {:ok, config}
      field -> {:error, "Missing required field: #{field}"}
    end
  end

  defp find_missing_field(config, fields) do
    Enum.find(fields, fn field ->
      not Map.has_key?(config, field)
    end)
  end

  @doc """
  Creates a new experiment configuration.

  ## Examples

      iex> {:ok, config} = CrucibleFramework.Experiment.new(
      ...>   name: "Test Experiment",
      ...>   conditions: ["baseline", "treatment"],
      ...>   metrics: [:accuracy, :latency]
      ...> )
      iex> config.name
      "Test Experiment"
  """
  @spec new(keyword()) :: {:ok, map()} | {:error, String.t()}
  def new(opts) when is_list(opts) do
    config = %{
      name: Keyword.get(opts, :name, "Unnamed Experiment"),
      description: Keyword.get(opts, :description, ""),
      conditions: Keyword.get(opts, :conditions, []),
      metrics: Keyword.get(opts, :metrics, []),
      repeat: Keyword.get(opts, :repeat, 1),
      seed: Keyword.get(opts, :seed, :os.system_time(:microsecond))
    }

    validate(config)
  end
end
