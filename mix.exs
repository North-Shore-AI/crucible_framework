defmodule CrucibleFramework.MixProject do
  use Mix.Project

  @version "0.5.1"
  @source_url "https://github.com/North-Shore-AI/crucible_framework"

  def project do
    [
      app: :crucible_framework,
      version: @version,
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      preferred_cli_env: [dialyzer: :dev],
      aliases: aliases(),
      deps: deps(),
      docs: docs(),
      description: description(),
      package: package(),
      name: "CrucibleFramework",
      source_url: @source_url,
      homepage_url: @source_url,
      dialyzer: [
        plt_add_apps: [:ex_unit, :mix],
        flags: [:error_handling, :underspecs]
      ]
    ]
  end

  def application do
    [
      mod: {CrucibleFramework.Application, []},
      extra_applications: [:logger, :telemetry]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Core IR (shared experiment definitions)
      {:crucible_ir, "~> 0.2.1"},

      # Reliability libraries (for built-in stage wrappers)
      # Made optional to break circular dependency with crucible_bench
      {:crucible_bench, "~> 0.4.0", optional: true},
      {:crucible_trace, "~> 0.3.0", optional: true},

      # Optional persistence
      {:ecto_sql, "~> 3.11", optional: true},
      {:postgrex, ">= 0.21.1", optional: true},

      # Core utilities
      {:jason, "~> 1.4"},
      {:telemetry, "~> 1.2"},

      # Development and Testing
      {:mox, "~> 1.1", only: :test},
      {:ex_doc, "~> 0.38", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp description do
    """
    CrucibleFramework: A thin orchestration layer for experiment pipelines.
    Provides pipeline execution, stage behaviour, and optional persistence.
    """
  end

  defp docs do
    [
      main: "readme",
      name: "CrucibleFramework",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      assets: %{"assets" => "assets"},
      logo: "assets/crucible_framework.svg",
      extras: extras(),
      groups_for_extras: groups_for_extras()
    ]
  end

  defp extras do
    [
      "README.md",
      "guides/getting_started.md",
      "guides/stages.md",
      "guides/configuration.md",
      "CHANGELOG.md",
      "LICENSE"
    ]
  end

  defp groups_for_extras do
    [
      Guides: [
        "guides/getting_started.md",
        "guides/stages.md",
        "guides/configuration.md"
      ]
    ]
  end

  defp package do
    [
      name: "crucible_framework",
      description: description(),
      files: ~w(README.md CHANGELOG.md mix.exs LICENSE lib guides assets),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/crucible_framework"
      },
      maintainers: ["nshkrdotcom"]
    ]
  end

  defp aliases do
    [
      setup: ["deps.get", "ecto.setup"],
      "ecto.setup": ["ecto.create", "ecto.migrate"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["test"]
    ]
  end
end
