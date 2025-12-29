defmodule CrucibleFramework do
  @moduledoc """
  Public entrypoints for running Crucible experiments.

  ## Database Configuration

  CrucibleFramework requires a Repo for persistence. Configure it in your host application:

      config :crucible_framework, repo: MyApp.Repo

  Then add CrucibleFramework to your supervision tree if you want managed persistence:

      children = [
        MyApp.Repo,
        # ... other children
      ]

  Run migrations using the provided mix task:

      mix crucible_framework.install

  Or copy migrations manually from `deps/crucible_framework/priv/repo/migrations/`.
  """

  alias Crucible.Pipeline.Runner
  alias CrucibleIR.Experiment

  @doc """
  Returns the configured Repo module.

  Raises if not configured. Configure with:

      config :crucible_framework, repo: MyApp.Repo
  """
  @spec repo() :: module()
  def repo do
    Application.get_env(:crucible_framework, :repo) ||
      raise ArgumentError, """
      CrucibleFramework requires a :repo configuration.

      Add to your config:

          config :crucible_framework, repo: MyApp.Repo
      """
  end

  @doc """
  Returns the configured Repo module, or nil if not configured.
  """
  @spec repo!() :: module() | nil
  def repo! do
    Application.get_env(:crucible_framework, :repo)
  end

  @spec run(Experiment.t(), keyword()) :: {:ok, Crucible.Context.t()} | {:error, term()}
  def run(%Experiment{} = experiment, opts \\ []) do
    Runner.run(experiment, opts)
  end
end
