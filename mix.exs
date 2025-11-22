defmodule CrucibleFramework.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/North-Shore-AI/crucible_framework"

  def project do
    [
      app: :crucible_framework,
      version: @version,
      elixir: "~> 1.14",
      preferred_cli_env: [dialyzer: :dev],
      deps: deps(),
      docs: docs(),
      description: description(),
      package: package(),
      name: "CrucibleFramework",
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  defp deps do
    [
      # Core dependency
      {:tinkex, "~> 0.1.1"},

      # Testing
      {:supertester, "~> 0.3.1", only: :test},
      {:mox, "~> 1.1", only: :test},
      {:stream_data, "~> 1.0", only: [:dev, :test]},

      # Documentation
      {:ex_doc, "~> 0.38", only: :dev, runtime: false},

      # Static analysis
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
      before_closing_head_tag: &mermaid_config/1
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
      "CHANGELOG.md"
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

  defp mermaid_config(:html) do
    """
    <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
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

  defp mermaid_config(_), do: ""

  defp package do
    [
      name: "crucible_framework",
      description: description(),
      files:
        ~w(README.md GETTING_STARTED.md ARCHITECTURE.md RESEARCH_METHODOLOGY.md ENSEMBLE_GUIDE.md HEDGING_GUIDE.md STATISTICAL_TESTING.md CAUSAL_TRANSPARENCY.md ADVERSARIAL_ROBUSTNESS.md INSTRUMENTATION.md DATASETS.md CONTRIBUTING.md FAQ.md PUBLICATIONS.md CHANGELOG.md mix.exs LICENSE lib assets),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/crucible_framework",
        "Component Libraries" => "https://github.com/North-Shore-AI"
      },
      maintainers: ["nshkrdotcom"]
    ]
  end
end
