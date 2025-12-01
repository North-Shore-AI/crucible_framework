defmodule CrucibleFramework do
  @moduledoc """
  Public entrypoints for running Crucible experiments.
  """

  alias CrucibleIR.Experiment
  alias Crucible.Pipeline.Runner

  @spec run(Experiment.t(), keyword()) :: {:ok, Crucible.Context.t()} | {:error, {atom(), term()}}
  def run(%Experiment{} = experiment, opts \\ []) do
    Runner.run(experiment, opts)
  end
end
