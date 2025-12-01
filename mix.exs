defmodule CrucibleFramework.MixProject do
  use Mix.Project

  @version "0.5.0"
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
      homepage_url: @source_url
    ]
  end

  def application do
    [
      mod: {CrucibleFramework.Application, []},
      extra_applications: [:logger, :crypto, :telemetry, :runtime_tools]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Shared IR
      {:crucible_ir, "~> 0.1.1"},

      # Component Libraries
      {:crucible_ensemble, path: "../crucible_ensemble"},
      {:crucible_hedging, path: "../crucible_hedging"},
      {:crucible_bench, path: "../crucible_bench"},
      {:crucible_trace, path: "../crucible_trace"},

      # Domain Libraries (optional - for full feature set)
      {:ex_fairness, path: "../ExFairness", optional: true},

      # Backend Integration
      {:tinkex, "~> 0.1.12"},

      # Core Dependencies
      {:ecto_sql, "~> 3.11"},
      {:postgrex, ">= 0.0.0"},
      {:jason, "~> 1.4"},
      {:telemetry, "~> 1.2"},
      {:nx, "~> 0.7"},

      # Development and Testing
      {:mox, "~> 1.1", only: :test},
      {:stream_data, "~> 1.0", only: [:dev, :test]},
      {:ex_doc, "~> 0.38", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
    ]
  end

  defp description do
    """
    CrucibleFramework: A scientific platform for LLM reliability research on the BEAM.
    Provides core library functionality with comprehensive documentation and guides.
    """
  end

  defp docs do
    [
      main: "readme",
      name: "CrucibleFramework",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      extras: extras(),
      groups_for_extras: groups_for_extras(),
      assets: %{"assets" => "assets"},
      logo: "assets/crucible_framework.svg",
      before_closing_head_tag: &mermaid_head/1,
      before_closing_body_tag: &mermaid_body/1
    ]
  end

  defp extras do
    [
      "README.md",
      "GETTING_STARTED.md",
      "ARCHITECTURE.md",
      "RESEARCH_METHODOLOGY.md",
      "ENSEMBLE_GUIDE.md",
      "HEDGING_GUIDE.md",
      "STATISTICAL_TESTING.md",
      "CAUSAL_TRANSPARENCY.md",
      "ADVERSARIAL_ROBUSTNESS.md",
      "INSTRUMENTATION.md",
      "DATASETS.md",
      "CONTRIBUTING.md",
      "FAQ.md",
      "PUBLICATIONS.md",
      "CHANGELOG.md",
      "LICENSE",
      "docs/tinkex_integration/00_architecture_overview.md",
      "docs/tinkex_integration/01_tinkex_adapter.md",
      "docs/tinkex_integration/02_lora_training_interface.md",
      "docs/tinkex_integration/03_ensemble_ml_integration.md",
      "docs/tinkex_integration/BUILD_PROMPT.md"
    ]
  end

  defp groups_for_extras do
    [
      "Getting Started": [
        "README.md",
        "GETTING_STARTED.md",
        "FAQ.md"
      ],
      "Architecture & Design": [
        "ARCHITECTURE.md",
        "RESEARCH_METHODOLOGY.md"
      ],
      "Component Guides": [
        "ENSEMBLE_GUIDE.md",
        "HEDGING_GUIDE.md",
        "STATISTICAL_TESTING.md",
        "CAUSAL_TRANSPARENCY.md",
        "ADVERSARIAL_ROBUSTNESS.md",
        "INSTRUMENTATION.md",
        "DATASETS.md"
      ],
      "Tinkex Integration": [
        "docs/tinkex_integration/00_architecture_overview.md",
        "docs/tinkex_integration/01_tinkex_adapter.md",
        "docs/tinkex_integration/02_lora_training_interface.md",
        "docs/tinkex_integration/03_ensemble_ml_integration.md",
        "docs/tinkex_integration/BUILD_PROMPT.md"
      ],
      Contributing: [
        "CONTRIBUTING.md"
      ],
      Research: [
        "PUBLICATIONS.md"
      ],
      "Release Notes": [
        "CHANGELOG.md"
      ]
    ]
  end

  defp mermaid_head(:html) do
    """
    <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
    """
  end

  defp mermaid_head(_), do: ""

  defp mermaid_body(:html) do
    """
    <script>
      let initialized = false;

      window.addEventListener("exdoc:loaded", () => {
        if (!initialized) {
          mermaid.initialize({
            startOnLoad: false,
            theme: document.body.className.includes("dark") ? "dark" : "default"
          });
          initialized = true;
        }

        let id = 0;
        for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
          const preEl = codeEl.parentElement;
          const graphDefinition = codeEl.textContent;
          const graphEl = document.createElement("div");
          const graphId = "mermaid-graph-" + id++;
          mermaid.render(graphId, graphDefinition).then(({svg, bindFunctions}) => {
            graphEl.innerHTML = svg;
            bindFunctions?.(graphEl);
            preEl.insertAdjacentElement("afterend", graphEl);
            preEl.remove();
          });
        }
      });
    </script>
    """
  end

  defp mermaid_body(_), do: ""

  defp package do
    [
      name: "crucible_framework",
      description: description(),
      files:
        ~w(README.md GETTING_STARTED.md ARCHITECTURE.md RESEARCH_METHODOLOGY.md ENSEMBLE_GUIDE.md HEDGING_GUIDE.md STATISTICAL_TESTING.md CAUSAL_TRANSPARENCY.md ADVERSARIAL_ROBUSTNESS.md INSTRUMENTATION.md DATASETS.md CONTRIBUTING.md FAQ.md PUBLICATIONS.md CHANGELOG.md mix.exs LICENSE lib assets docs),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/crucible_framework",
        "Component Libraries" => "https://github.com/North-Shore-AI"
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
