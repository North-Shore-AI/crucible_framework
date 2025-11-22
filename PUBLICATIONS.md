# Publications Guide

This document provides comprehensive guidelines for citing CrucibleFramework in academic publications, sharing research results, and maintaining reproducibility standards.

**Table of Contents**

- [How to Cite This Framework](#how-to-cite-this-framework)
- [BibTeX Entries](#bibtex-entries)
- [Paper Templates](#paper-templates)
- [Methods Section Templates](#methods-section-templates)
- [Results Section Templates](#results-section-templates)
- [Dataset Sharing Protocols](#dataset-sharing-protocols)
- [Reproducibility Requirements](#reproducibility-requirements)
- [Example Acknowledgments](#example-acknowledgments)
- [Open Science Practices](#open-science-practices)
- [Publishing Checklist](#publishing-checklist)

---

## How to Cite This Framework

### Citation Formats

**General Framework Citation:**

When using the framework in general, cite:

```
North Shore AI. (2025). CrucibleFramework: Infrastructure for LLM Reliability Research
(Version 0.1.4) [Computer software]. https://github.com/North-Shore-AI/crucible_framework
```

**APA Style:**
```
North Shore AI. (2025). CrucibleFramework: Infrastructure for LLM Reliability Research
(Version 0.1.4) [Computer software]. GitHub. https://github.com/North-Shore-AI/crucible_framework
```

**IEEE Style:**
```
North Shore AI, "CrucibleFramework: Infrastructure for LLM Reliability Research,"
version 0.1.4, 2025. [Online]. Available:
https://github.com/North-Shore-AI/crucible_framework
```

**MLA Style:**
```
North Shore AI. CrucibleFramework: Infrastructure for LLM Reliability Research.
Version 0.1.4, GitHub, 2025, github.com/North-Shore-AI/crucible_framework.
```

### Citing Specific Libraries

When using a specific library extensively, also cite that library:

**Ensemble Library:**
```
North Shore AI. (2025). CrucibleFramework Ensemble Module: Multi-model voting for AI reliability
(Version 0.1.4) [Computer software]. Part of CrucibleFramework.
https://github.com/North-Shore-AI/crucible_framework
```

**Hedging Library:**
```
North Shore AI. (2025). CrucibleFramework Hedging Module: Request hedging for tail latency reduction
(Version 0.1.4) [Computer software]. Part of CrucibleFramework.
https://github.com/North-Shore-AI/crucible_framework
```

**Bench Library:**
```
North Shore AI. (2025). CrucibleFramework Bench Module: Statistical testing for AI research
(Version 0.1.4) [Computer software]. Part of CrucibleFramework.
https://github.com/North-Shore-AI/crucible_framework
```

### In-Text Citations

**First mention:**
```
We used CrucibleFramework (North Shore AI, 2025) to conduct our experiments on ensemble reliability.
```

**Subsequent mentions:**
```
The framework's ensemble module provides four voting strategies...
```

---

## BibTeX Entries

### Main Framework

```bibtex
@software{crucible_framework2025,
  title = {CrucibleFramework: Infrastructure for LLM Reliability Research},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {A scientifically rigorous infrastructure for LLM reliability and performance research}
}
```

### Ensemble Library

```bibtex
@software{crucible_framework_ensemble2025,
  title = {CrucibleFramework Ensemble Module: Multi-model Voting for AI Reliability},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {ensemble learning, voting strategies, LLM reliability}
}
```

### Hedging Library

```bibtex
@software{crucible_framework_hedging2025,
  title = {CrucibleFramework Hedging Module: Request Hedging for Tail Latency Reduction},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework. Implements techniques from Dean \& Barroso (2013)},
  keywords = {tail latency, request hedging, distributed systems}
}
```

### Bench Library

```bibtex
@software{crucible_framework_bench2025,
  title = {CrucibleFramework Bench Module: Statistical Testing for AI Research},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {statistical testing, effect sizes, reproducibility}
}
```

### TelemetryResearch Library

```bibtex
@software{crucible_framework_telemetry2025,
  title = {CrucibleFramework Telemetry Module: Research-grade Instrumentation},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {instrumentation, telemetry, experiment tracking}
}
```

### DatasetManager Library

```bibtex
@software{crucible_framework_dataset2025,
  title = {CrucibleFramework Dataset Module: Unified Benchmark Dataset Interface},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {benchmarks, datasets, evaluation}
}
```

### CausalTrace Library

```bibtex
@software{crucible_framework_causal2025,
  title = {CrucibleFramework CausalTrace Module: Decision Provenance for LLMs},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {transparency, provenance, explainability}
}
```

### ResearchHarness Library

```bibtex
@software{crucible_framework_orchestration2025,
  title = {CrucibleFramework Orchestration Module: Experiment DSL},
  author = {{North Shore AI}},
  year = {2025},
  month = {11},
  url = {https://github.com/North-Shore-AI/crucible_framework},
  version = {0.1.4},
  note = {Part of CrucibleFramework},
  keywords = {experiment design, orchestration, reproducibility}
}
```

### Related Works to Cite

**Ensemble Methods:**

```bibtex
@article{breiman1996bagging,
  title={Bagging predictors},
  author={Breiman, Leo},
  journal={Machine learning},
  volume={24},
  number={2},
  pages={123--140},
  year={1996},
  publisher={Springer}
}

@article{dietterich2000ensemble,
  title={Ensemble methods in machine learning},
  author={Dietterich, Thomas G},
  journal={International workshop on multiple classifier systems},
  pages={1--15},
  year={2000},
  organization={Springer}
}
```

**Tail Latency:**

```bibtex
@article{dean2013tail,
  title={The tail at scale},
  author={Dean, Jeffrey and Barroso, Luiz Andr{\'e}},
  journal={Communications of the ACM},
  volume={56},
  number={2},
  pages={74--80},
  year={2013},
  publisher={ACM New York, NY, USA}
}
```

**Statistical Methods:**

```bibtex
@book{cohen1988statistical,
  title={Statistical power analysis for the behavioral sciences},
  author={Cohen, Jacob},
  year={1988},
  publisher={Routledge}
}

@article{cumming2014new,
  title={The new statistics: Why and how},
  author={Cumming, Geoff},
  journal={Psychological science},
  volume={25},
  number={1},
  pages={7--29},
  year={2014},
  publisher={Sage Publications}
}
```

**Benchmarks:**

```bibtex
@article{hendrycks2021mmlu,
  title={Measuring massive multitask language understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{chen2021humaneval,
  title={Evaluating large language models trained on code},
  author={Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and others},
  journal={arXiv preprint arXiv:2107.03374},
  year={2021}
}

@article{cobbe2021gsm8k,
  title={Training verifiers to solve math word problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

---

## Paper Templates

### Abstract Template

```latex
\begin{abstract}
We investigate [research question] using [experimental design].
We employ CrucibleFramework \cite{crucible_framework2025}
to conduct [number] experiments with [number] queries across [number] conditions.
Our results show that [main finding] with [statistical significance].
Specifically, [treatment] achieved [metric value] compared to [baseline value],
representing a [percentage]% improvement (p < [p-value], d = [effect size]).
These findings have implications for [practical application].
All code and data are available at [repository URL].
\end{abstract}
```

**Example:**

```latex
\begin{abstract}
We investigate the reliability of multi-model ensembles for mathematical reasoning
using a randomized controlled trial. We employ CrucibleFramework
\cite{crucible_framework2025} to conduct 3 experiments with 200 queries across
4 ensemble sizes (1, 3, 5, 7 models). Our results show that 5-model ensembles
significantly outperform single models with very large effect sizes. Specifically,
the 5-model ensemble achieved 96.3\% accuracy (SD=1.2\%) compared to 89.1\% (SD=2.1\%)
for single model baseline, representing a 7.2 percentage point improvement
(p < 0.001, d = 3.42). These findings have implications for deploying reliable
AI systems in production environments. All code and data are available at
https://osf.io/xxxxx/.
\end{abstract}
```

### Introduction Template

```latex
\section{Introduction}

[Context and motivation for the research]

Recent advances in large language models (LLMs) have enabled remarkable capabilities,
yet reliability remains a critical challenge \cite{relevant_citation}.
[Specific problem statement].

[Gap in existing research]

While prior work has explored [related approach], limited research has examined
[your specific focus]. This gap is significant because [importance].

[Your contribution]

In this paper, we address this gap by [your approach]. We make the following
contributions:

\begin{itemize}
    \item We propose [contribution 1]
    \item We demonstrate [contribution 2] using rigorous experimental methodology
    \item We provide [contribution 3], enabling reproducibility
\end{itemize}

[Overview of paper structure]

The remainder of this paper is organized as follows. Section \ref{sec:related}
reviews related work. Section \ref{sec:methods} describes our experimental
methodology. Section \ref{sec:results} presents results. Section \ref{sec:discussion}
discusses implications, and Section \ref{sec:conclusion} concludes.
```

---

## Methods Section Templates

### Experimental Design Template

```latex
\section{Methods}
\label{sec:methods}

\subsection{Experimental Design}

We conducted a [design type, e.g., randomized controlled trial] to test the
hypothesis that [hypothesis statement]. The independent variable was
[IV description] with [number] levels: [level descriptions]. The dependent
variable was [DV description], measured as [measurement method].

\textbf{Participants/Queries:} We sampled [number] queries from [dataset]
\cite{dataset_citation}. [Sampling method and justification].

\textbf{Conditions:} We implemented [number] experimental conditions:

\begin{itemize}
    \item \textbf{Baseline:} [Description]
    \item \textbf{Treatment 1:} [Description]
    \item \textbf{Treatment 2:} [Description]
\end{itemize}

\textbf{Procedure:} For each query in each condition, we [procedure description].
We repeated the entire experiment [number] times to account for stochastic
variation in model outputs and API latencies.

\textbf{Randomization:} Query order was randomized using a seeded random number
generator (seed=42) to ensure reproducibility while preventing order effects.

\subsection{Implementation}

We implemented all experiments using CrucibleFramework
\cite{crucible_framework2025}, which provides research-grade instrumentation
and statistical analysis capabilities.

\textbf{Ensemble Implementation:} We used the CrucibleFramework Ensemble module
\cite{crucible_framework_ensemble2025} to implement multi-model voting with majority strategy.
Models were queried in parallel with a timeout of [timeout] seconds.

\textbf{Statistical Analysis:} We used the CrucibleFramework Bench module \cite{crucible_framework_bench2025}
for statistical testing. We employed [test name] to compare conditions, with
$\alpha = 0.05$ as the significance threshold. Effect sizes were calculated
using [effect size measure] following Cohen (1988) \cite{cohen1988statistical}.

\textbf{Hardware:} Experiments were conducted on [hardware description].

\textbf{Cost:} Total experiment cost was approximately \$[cost], with
[per-query cost] per query.

\subsection{Metrics}

We measured the following metrics:

\begin{itemize}
    \item \textbf{Accuracy:} Percentage of correct predictions
    \item \textbf{Latency:} Time from query submission to response (P50, P95, P99)
    \item \textbf{Cost:} Total API cost per query in USD
    \item \textbf{Consensus:} Agreement rate among ensemble members (0-1)
\end{itemize}

\subsection{Reproducibility}

To ensure reproducibility, we:

\begin{enumerate}
    \item Fixed random seeds for all stochastic operations
    \item Recorded exact model versions (GPT-4: gpt-4-0613, Claude: claude-3-opus-20240229)
    \item Saved all experimental artifacts (configurations, raw results, analysis scripts)
    \item Published code and data at [repository URL]
\end{enumerate}

All experiments are reproducible using the command:

\begin{verbatim}
mix run experiments/h1_ensemble.exs --seed 42
\end{verbatim}
```

### Statistical Analysis Template

```latex
\subsection{Statistical Analysis}

\textbf{Hypothesis Testing:} We tested the null hypothesis that [H0 description]
against the alternative hypothesis that [H1 description]. Given [data characteristics],
we used [test name] with [assumptions description].

\textbf{Sample Size:} We determined sample size using a priori power analysis.
To detect a [effect size] effect with power = [power] and $\alpha = 0.05$,
we required $n \geq [n]$ per condition. We collected $n = [actual n]$ to
ensure adequate power.

\textbf{Multiple Comparisons:} When comparing more than two conditions, we
applied [correction method] correction to control family-wise error rate.

\textbf{Effect Sizes:} We report Cohen's $d$ for all pairwise comparisons,
interpreting $d \geq 0.2$ as small, $d \geq 0.5$ as medium, and $d \geq 0.8$
as large effects \cite{cohen1988statistical}.

\textbf{Confidence Intervals:} We report 95\% confidence intervals for all
effect size estimates using [bootstrap/parametric] methods.

\textbf{Assumptions:} We tested statistical assumptions using [tests]:
\begin{itemize}
    \item Normality: [test] ($p = [p-value]$)
    \item Homogeneity of variance: [test] ($p = [p-value]$)
    \item Independence: Verified by experimental design
\end{itemize}

When assumptions were violated, we used [non-parametric alternative].
```

---

## Results Section Templates

### Descriptive Statistics Template

```latex
\section{Results}
\label{sec:results}

\subsection{Descriptive Statistics}

Table \ref{tab:descriptive} presents descriptive statistics for all experimental
conditions. [Condition name] achieved the highest accuracy ($M = [mean]$,
$SD = [sd]$), followed by [other conditions].

\begin{table}[h]
\centering
\caption{Descriptive statistics by experimental condition}
\label{tab:descriptive}
\begin{tabular}{lcccc}
\toprule
Condition & $M$ & $SD$ & $Min$ & $Max$ \\
\midrule
Baseline & 89.1 & 2.1 & 85.5 & 92.0 \\
Treatment 1 & 94.3 & 1.5 & 91.8 & 96.5 \\
Treatment 2 & 96.3 & 1.2 & 94.2 & 98.0 \\
\bottomrule
\end{tabular}
\end{table}

Figure \ref{fig:distribution} shows the distribution of accuracy scores across
conditions. [Description of patterns].
```

### Inferential Statistics Template

```latex
\subsection{Inferential Statistics}

We conducted [test name] to compare accuracy across conditions. Results showed
a statistically significant effect of condition on accuracy,
$F([df1], [df2]) = [F-value]$, $p < [p-value]$, $\eta^2 = [effect-size]$.

Post-hoc pairwise comparisons using [correction method] revealed:

\begin{itemize}
    \item Treatment 1 vs. Baseline: $t([df]) = [t-value]$, $p < [p-value]$,
          $d = [effect-size]$ (95\% CI: [[lower], [upper]])
    \item Treatment 2 vs. Baseline: $t([df]) = [t-value]$, $p < [p-value]$,
          $d = [effect-size]$ (95\% CI: [[lower], [upper]])
    \item Treatment 2 vs. Treatment 1: $t([df]) = [t-value]$, $p = [p-value]$,
          $d = [effect-size]$ (95\% CI: [[lower], [upper]])
\end{itemize}

All pairwise comparisons except [exception] reached statistical significance
at the Bonferroni-corrected $\alpha = [adjusted-alpha]$ level.

\subsection{Effect Sizes}

Effect sizes were large for all comparisons. Treatment 2 showed a very large
effect relative to baseline ($d = 3.42$), indicating practical significance
in addition to statistical significance.
```

### Cost-Benefit Analysis Template

```latex
\subsection{Cost-Benefit Analysis}

While Treatment 2 achieved highest accuracy, it also incurred highest cost
(Table \ref{tab:cost}). We calculated cost per percentage point improvement
as:

\begin{equation}
    \text{Cost Efficiency} = \frac{\text{Cost}}{\text{Accuracy Improvement}}
\end{equation}

Treatment 1 offered the best cost-efficiency at \$0.42 per percentage point,
compared to \$1.03 for Treatment 2.

\begin{table}[h]
\centering
\caption{Cost-benefit analysis}
\label{tab:cost}
\begin{tabular}{lccc}
\toprule
Condition & Cost & Accuracy Gain & Cost/Point \\
\midrule
Baseline & \$0.01 & - & - \\
Treatment 1 & \$0.03 & 5.2\% & \$0.42 \\
Treatment 2 & \$0.09 & 7.2\% & \$1.03 \\
\bottomrule
\end{tabular}
\end{table}

For production deployments, Treatment 1 offers an optimal balance of reliability
improvement and cost.
```

---

## Dataset Sharing Protocols

### Dataset Documentation

When sharing datasets, include comprehensive documentation:

```markdown
# Dataset Name

## Overview

- **Name:** [Dataset name]
- **Version:** [Semantic version]
- **Release Date:** [Date]
- **License:** [License type]
- **DOI:** [DOI if registered]

## Description

[Paragraph describing dataset purpose, contents, and use cases]

## Collection Methodology

- **Collection Method:** [How data was collected]
- **Collection Period:** [Time period]
- **Collection Tools:** [Software/hardware used]
- **Quality Control:** [QC procedures applied]

## Format

- **File Format:** JSONL (JSON Lines)
- **Encoding:** UTF-8
- **Compression:** gzip

## Schema

```json
{
  "id": "unique_identifier",
  "query": "Question or prompt text",
  "expected": "Ground truth answer",
  "metadata": {
    "source": "Source of query",
    "difficulty": "easy|medium|hard",
    "domain": "Subject domain"
  }
}
```

## Statistics

- **Total Examples:** [Number]
- **Unique Domains:** [Number]
- **Average Query Length:** [Number] tokens
- **Answer Distribution:** [Description]

## Usage

Load your dataset:

```elixir
dataset = DatasetManager.load(:your_dataset)
```

## Citation

```bibtex
@dataset{your_dataset2025,
  title = {Dataset Name},
  author = {Your Name},
  year = {2025},
  publisher = {Platform},
  doi = {10.xxxx/xxxxx}
}
```

## Ethical Considerations

- **Privacy:** [PII removal procedures]
- **Bias:** [Known biases and limitations]
- **Terms of Use:** [Usage restrictions]

## Contact

- **Maintainer:** [Name]
- **Email:** [Email]
- **Repository:** [URL]
```

### File Naming Convention

```
dataset_name_v1.0.0_train.jsonl.gz
dataset_name_v1.0.0_test.jsonl.gz
dataset_name_v1.0.0_validation.jsonl.gz
dataset_name_v1.0.0_metadata.json
dataset_name_v1.0.0_README.md
```

---

## Reproducibility Requirements

### Computational Environment

Document exact computational environment:

```yaml
# environment.yml
framework:
  name: crucible_framework
  version: 0.1.4
  commit: v0.1.4

runtime:
  elixir: 1.17.0
  erlang: 26.2.1
  os: Ubuntu 22.04 LTS

models:
  gpt4:
    name: gpt-4
    version: gpt-4-0613
    api_date: 2025-10-08
  claude:
    name: claude-3-opus
    version: claude-3-opus-20240229
    api_date: 2025-10-08

hardware:
  cpu: Intel Xeon E5-2670
  memory: 64GB
  gpu: None

configuration:
  seed: 42
  sample_size: 200
  repetitions: 3
  timeout_ms: 30000
```

### Artifact Checklist

**Minimum artifacts for reproducibility:**

- [ ] **Source code:** All experiment code
- [ ] **Configuration files:** Exact settings used
- [ ] **Dataset:** Either dataset files or download instructions
- [ ] **Results:** Raw results in CSV/JSON format
- [ ] **Analysis scripts:** Statistical analysis code
- [ ] **Environment specification:** Dependencies and versions
- [ ] **README:** Instructions for reproduction
- [ ] **License:** Clear licensing for reuse

### Repository Structure

```
experiment_name/
├── README.md                    # Reproduction instructions
├── environment.yml              # Computational environment
├── config/
│   └── experiment_config.exs   # Experiment configuration
├── data/
│   ├── dataset_v1.0.0.jsonl    # Dataset (or download script)
│   └── metadata.json           # Dataset metadata
├── experiments/
│   └── run_experiment.exs      # Experiment script
├── results/
│   ├── raw_results.csv         # Raw experimental results
│   ├── analysis.json           # Statistical analysis output
│   └── figures/                # Generated figures
├── analysis/
│   └── analyze.exs             # Analysis scripts
└── LICENSE                      # License file
```

---

## Example Acknowledgments

### General Framework Usage

```latex
\section*{Acknowledgments}

We thank the contributors to CrucibleFramework for providing
the infrastructure that enabled this research. Experiments were conducted
using version 0.1.4 of the framework.
```

### Specific Library Usage

```latex
\section*{Acknowledgments}

We thank the contributors to CrucibleFramework. This work
specifically utilized the Ensemble module for multi-model voting and the
Bench module for statistical analysis. We also acknowledge [funding source]
for supporting this research.
```

### Community Contributions

```latex
\section*{Acknowledgments}

We thank [contributor names] for helpful discussions and feedback on
experimental design. We thank the CrucibleFramework community
for providing tools and infrastructure. This work was supported by [funding].
```

### With Co-authors from Framework Team

```latex
\section*{Author Contributions}

[Your name] conceived the study, designed experiments, and conducted analysis.
[Framework author] provided technical guidance on framework usage and reviewed
statistical methods. [Other authors] contributed to [contributions]. All
authors reviewed and approved the final manuscript.
```

---

## Open Science Practices

### Pre-registration

Consider pre-registering your study:

**Platforms:**
- Open Science Framework (OSF): https://osf.io/
- AsPredicted: https://aspredicted.org/
- ClinicalTrials.gov (for clinical studies)

**What to pre-register:**
- Research questions and hypotheses
- Experimental design and sample size
- Statistical analysis plan
- Primary and secondary outcomes
- Data collection stopping rules

**Example OSF registration:**

```markdown
# Pre-registration: Ensemble Reliability Study

## Research Question

Does a 5-model ensemble achieve ≥99% accuracy on MMLU-STEM?

## Hypotheses

H1: 5-model ensemble accuracy > single model accuracy (one-tailed, α=0.05)

## Design

Randomized controlled trial, n=200 queries, 3 repetitions

## Statistical Analysis

Primary: Independent t-test comparing ensemble vs. baseline
Effect size: Cohen's d with 95% CI
Power: 0.80 to detect d=0.5

## Data Collection

Stop after 3 complete repetitions, regardless of results
```

### Data Sharing

**Where to share:**

- **GitHub:** Code and small datasets
- **Zenodo:** Large datasets and long-term archival (DOI)
- **OSF:** Complete project including pre-registration
- **Hugging Face Datasets:** ML datasets
- **Figshare:** Figures and supplementary materials

**Licensing:**

- **Code:** MIT
- **Data:** CC0 (public domain), CC-BY (attribution), or CC-BY-SA (share-alike)
- **Papers:** CC-BY for gold open access

### Open Access Publication

Consider publishing in open access journals:

- JMLR (Journal of Machine Learning Research)
- JAIR (Journal of Artificial Intelligence Research)
- NeurIPS, ICML, ACL (via proceedings)
- arXiv preprints

---

## Publishing Checklist

### Before Submission

- [ ] **Pre-registration** (optional but recommended)
- [ ] **IRB approval** (if human subjects)
- [ ] **Data use agreements** (for proprietary datasets)
- [ ] **Co-author approval** from all authors

### Manuscript Preparation

- [ ] **Title** accurately reflects content
- [ ] **Abstract** follows journal guidelines
- [ ] **Keywords** selected appropriately
- [ ] **Introduction** motivates the research
- [ ] **Methods** sufficient for reproduction
- [ ] **Results** clearly presented with statistics
- [ ] **Discussion** interprets findings
- [ ] **Conclusion** summarizes contributions
- [ ] **References** complete and formatted correctly
- [ ] **Figures** high resolution and properly labeled
- [ ] **Tables** formatted per journal style

### Supplementary Materials

- [ ] **Appendix** with additional details
- [ ] **Code repository** link included
- [ ] **Data repository** link included
- [ ] **Reproduction instructions** provided
- [ ] **Ethics statement** included (if required)

### After Acceptance

- [ ] **Upload preprint** to arXiv
- [ ] **Share on social media** and mailing lists
- [ ] **Update repository** with published citation
- [ ] **Share code/data** per data availability statement
- [ ] **Present findings** at conferences
- [ ] **Engage with community** responses

---

## Frequently Asked Questions

### When should I cite the framework?

Cite the main framework if you use any of its libraries. Additionally cite
specific libraries you use extensively.

### Can I modify the framework for my research?

Yes! The framework is open source (MIT license). If you make modifications,
please:
1. Document your changes clearly
2. Indicate that you used a modified version
3. Consider contributing improvements back

### Should I cite the framework in the abstract?

Generally no. Cite in the Methods section where you describe your implementation.
Mention in the abstract only if the framework itself is a contribution of the paper.

### How do I handle framework updates during research?

1. Pin to a specific version at start of research
2. Document the version number
3. Don't update mid-study unless fixing a critical bug
4. If you must update, note this and re-run validation checks

### What if I can't share my dataset?

If dataset is proprietary:
1. Describe it thoroughly in the methods
2. Share aggregated statistics
3. Provide code for reproduction with public datasets
4. Consider releasing a subset or synthetic version

---

## Additional Resources

### Style Guides

- **APA 7th Edition:** https://apastyle.apa.org/
- **IEEE:** https://ieeeauthorcenter.ieee.org/
- **ACM:** https://www.acm.org/publications/authors/reference-formatting

### Reproducibility Resources

- **Papers with Code:** https://paperswithcode.com/
- **ReScience Journal:** https://rescience.github.io/
- **Nature Reproducibility Guidelines:** https://www.nature.com/nature-research/editorial-policies/reporting-standards

### Statistical Reporting

- **APA Statistical Reporting:** https://apastyle.apa.org/instructional-aids/numbers-statistics-guide.pdf
- **JARS (Journal Article Reporting Standards):** https://apastyle.apa.org/jars

---

**Last Updated:** 2025-11-21
**Version:** 0.1.4
**Maintainers:** North Shore AI

---

Built with ❤️ by researchers, for researchers.
