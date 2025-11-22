# Tinkex Integration Plan for Crucible Framework

**Version:** 1.0
**Date:** 2025-11-21
**Authors:** North-Shore-AI Team
**Status:** Proposal

---

## 1. Executive Summary

### Why Integrate Tinkex with Crucible

Tinkex is the Elixir SDK for the Tinker ML Training and Inference API, providing a functional, concurrent interface for fine-tuning large language models using LoRA (Low-Rank Adaptation) and performing high-performance text generation. Integrating Tinkex with Crucible Framework creates a powerful, end-to-end research infrastructure for LLM reliability experiments.

#### Strategic Value

1. **Unified Experiment Orchestration**: Crucible's `crucible_harness` DSL can orchestrate complete training-evaluation cycles, with Tinkex handling the remote GPU execution and model management.

2. **Statistical Rigor for Fine-Tuning**: Crucible's `crucible_bench` provides 15+ statistical tests for evaluating fine-tuned model performance with publication-quality results.

3. **Multi-Model Ensembles from Fine-Tuned Variants**: `crucible_ensemble` can coordinate voting across multiple LoRA adapters trained with different hyperparameters.

4. **Research-Grade Telemetry**: Connect Tinkex's telemetry events to `crucible_telemetry` for complete experimental traceability and reproducibility.

5. **Latency Optimization**: Leverage `crucible_hedging` strategies for reducing P99 tail latencies during inference with Tinkex SamplingClients.

6. **Native Elixir/OTP Architecture**: Both Tinkex and Crucible are built on Elixir/OTP, enabling seamless process supervision, fault tolerance, and concurrent operations.

#### Expected Outcomes

- 50-75% reduction in experiment iteration time through automated orchestration
- Research-grade metrics collection enabling publication-quality analysis
- Reproducible experiments with complete provenance tracking
- Scalable multi-model evaluation across distributed GPU resources

---

## 2. Lessons from Thinker Experiments

The CNS3 (CNS Support Models) project has conducted extensive experiments using the Thinker orchestration layer with Tinker backend. These experiments provide critical insights for the Crucible integration.

### 2.1 Training Iterations with citation_validity_weight

The CNS3 team trained claim extractor models on SciFact data with varying penalty weights for citation validity. Key findings:

#### Experiment Configuration
- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Dataset**: SciFact (505 examples)
- **Training**: 5 epochs, 320 steps total
- **LoRA Configuration**: rank=16, alpha=32, dropout=0.05
- **Optimizer**: AdamW, learning_rate=0.0002
- **Batch size**: 8

#### Weight Iteration Results

| Weight | Loss Reduction | Citation Invalid Rate | Outcome |
|--------|---------------|----------------------|---------|
| 1.0 (baseline) | N/A | N/A | Baseline performance |
| 2.0 (3x multiplier) | 98.7% | 0.000 | Training data clean, but 2 HIGH severity flags persist |
| 5.0 (recommended) | Pending | Pending | Expected to reduce hallucinations |

#### Key Insight

The citation validation mechanism worked correctly during training (citation_invalid_rate=0.000), but the model did not generalize to avoid hallucinating citations during inference. This demonstrates the need for:

- **Negative Examples**: Training data should include examples of invalid citations with high penalties
- **Stronger Penalties**: citation_validity_weight=5.0 or higher for meaningful impact
- **Evaluation-Driven Iteration**: Tight feedback loops between training and evaluation

### 2.2 Antagonist MVP Results

The Antagonist component evaluates model outputs for quality issues using semantic and topological analysis.

#### Flagging Performance
- **Total Flags**: 46/50 samples (92% flagging rate)
- **HIGH Severity**: 2 flags (CITATION_INVALID)
- **MEDIUM Severity**: 44 flags (WEAK_ENTAILMENT, POLARITY_CONTRADICTION)

#### HIGH Severity Analysis (Claims 133 & 179)

```json
{
  "claim_id": 133,
  "severity": "HIGH",
  "issues": [
    {"issue_type": "CITATION_INVALID", "details": {"reason": "Model cited documents not present in source corpus"}},
    {"issue_type": "POLARITY_CONTRADICTION", "details": {"chirality_score": 0.735}},
    {"issue_type": "WEAK_ENTAILMENT", "details": {"entailment_score": 0.0}}
  ]
}
```

#### Antagonist Triggers
- **Polarity Stress Tests**: Triggered when chirality_score >= 0.55 or polarity_conflict == true
- **Evidence Consistency**: Flags when entailment_score < 0.5
- **Chirality Delta Coverage**: Target >= 0.9 for high-chirality cases

### 2.3 Quality Metrics and Targets

Based on CNS3 experiments, the following quality targets are established:

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Schema Compliance | 98-100% | >= 95% | Must maintain structural correctness |
| Citation Accuracy | 96% | >= 95% | Valid citations only |
| Mean Entailment | 0.395-0.448 | >= 0.50 | Claims must be supported by evidence |
| Entailment Pass Rate | 34% | >= 45% | Threshold: >= 0.75 entailment |
| Semantic Pass Rate | 34-38% | >= 45% | Combined semantic quality |
| Similarity Pass Rate | 18% | >= 35% | Threshold: >= 0.70 similarity |

### 2.4 Jacobian Mastery Insights for LoRA Rank Selection

The CNS3 experiments explored LoRA rank selection using Jacobian analysis for understanding gradient flow:

#### LoRA Configuration Rationale
- **Rank 16**: Selected for claim extraction task (lower complexity)
- **Rank 32**: Standard for general fine-tuning
- **Alpha/Rank Ratio**: 2x (alpha=32 for rank=16)

#### Per-Layer Rank Analysis (Declined)

A Tinker feature request for per-layer LoRA rank configuration was evaluated but declined due to:

1. **Complexity vs. Benefit**: Marginal gains don't justify implementation complexity
2. **Research Priority**: Focus on end-to-end quality metrics first
3. **Tinker Compatibility**: Service API currently supports uniform rank only

The decision aligns with the practical finding that uniform rank=16 with proper loss weighting achieves target metrics.

### 2.5 Topology Instrumentation Results

The CNS3 experiments added topology analysis to evaluation:

- **Beta_1 (Betti Number)**: 0 across all 50 samples (graphs are DAGs, no cycles)
- **Mean Chirality Score**: 0.561 (healthy tension even without cycles)
- **Mean Fisher-Rao Distance**: 16.75

This indicates the Antagonist must focus on chirality/polarity contradictions rather than cycle detection.

---

## 3. Tinkex Architecture Analysis

Tinkex follows Elixir/OTP conventions with a well-structured client hierarchy.

### 3.1 Client Hierarchy

```
ServiceClient (Entry Point)
    +-- TrainingClient (per model instance)
    |   +-- SamplingClient (from saved weights)
    +-- SamplingClient (standalone inference)
    +-- RestClient (low-level API operations)
```

### 3.2 TrainingClient GenServer Design

The TrainingClient manages model training operations for a specific model instance.

#### Key Features
- **Request Sequencing**: Operations execute in request ID order
- **Data Chunking**: Large batches split into chunks (max 128 examples, max 500K numbers)
- **Synchronous Sends with Async Polling**: Requests sent sequentially, results polled concurrently

#### Implementation Pattern

```elixir
defmodule Tinkex.TrainingClient do
  use GenServer

  @max_chunk_len 128
  @max_chunk_number_count 500_000

  def forward_backward(client, data, loss_fn, opts \\ []) do
    GenServer.call(client, {:forward_backward, data, loss_fn, opts}, :infinity)
  end

  @impl true
  def handle_call({:forward_backward, data, loss_fn, opts}, from, state) do
    chunks = chunk_data(data)
    {request_ids, new_counter} = allocate_request_ids(length(chunks), state.request_id_counter)

    # Send ALL requests SYNCHRONOUSLY to ensure ordering
    send_result = Enum.reduce_while(Enum.zip(request_ids, chunks), {:ok, []}, ...)

    case send_result do
      {:ok, untyped_futures} ->
        # Spawn background task for concurrent polling
        Task.start(fn ->
          results = Task.await_many(polling_tasks, :infinity)
          GenServer.reply(from, {:ok, combine_results(results)})
        end)
        {:noreply, %{state | request_id_counter: new_counter}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end
end
```

#### Responsiveness Trade-off

The TrainingClient blocks during the synchronous send phase. This is acceptable for v1.0 because:
- Training operations are sequential by design
- Blocking ensures strict request ordering
- Matches Python SDK behavior

### 3.3 SamplingClient Lock-Free ETS Pattern

The SamplingClient uses ETS for lock-free reads to handle high concurrency (400+ concurrent requests).

#### Architecture Benefits
- **Zero GenServer.call bottleneck**: ETS reads are ~100ns vs ~5-10us for GenServer.call
- **Lock-free counters**: Atomics for request IDs and backoff timestamps
- **Shared rate limiting**: Per {base_url, api_key} combination

#### Implementation Pattern

```elixir
defmodule Tinkex.SamplingClient do
  def sample(client, prompt, num_samples, sampling_params, opts \\ []) do
    Task.async(fn ->
      # Read config from ETS (lock-free)
      case :ets.lookup(:tinkex_sampling_clients, {:config, client}) do
        [{_, config}] ->
          # Wait for any active backoff
          Tinkex.RateLimiter.wait_for_backoff(config.rate_limiter)

          # Atomic increment for request ID
          request_id = :atomics.add_get(config.request_id_counter, 1, 1)

          # HTTP call in caller's process
          Tinkex.API.Sampling.asample(request, config.http_pool, api_opts)

        [] ->
          {:error, %Tinkex.Error{message: "SamplingClient not initialized"}}
      end
    end)
  end
end
```

#### Performance Comparison
- **GenServer.call**: 400 requests x 5us = 2ms serialization overhead
- **ETS approach**: 400 requests x 100ns = 40us total overhead
- **Speedup**: 50x faster at high concurrency

### 3.4 Task-Based Futures for Async Operations

Tinkex returns Elixir Tasks for all async operations, enabling:

```elixir
# Single operation
{:ok, task} = Tinkex.TrainingClient.forward_backward(client, data, :cross_entropy)
{:ok, result} = Task.await(task)

# Concurrent operations
tasks = Enum.map(prompts, fn prompt ->
  {:ok, task} = Tinkex.SamplingClient.sample(sampler, prompt, params)
  task
end)
results = Task.await_many(tasks, 5_000)
```

### 3.5 Rate Limiting and Retry Strategies

#### RateLimiter Module

```elixir
defmodule Tinkex.RateLimiter do
  def for_key({base_url, api_key}) do
    normalized = Tinkex.PoolKey.normalize_base_url(base_url)
    key = {:limiter, {normalized, api_key}}

    limiter = :atomics.new(1, signed: true)
    case :ets.insert_new(:tinkex_rate_limiters, {key, limiter}) do
      true -> limiter
      false ->
        [{^key, existing}] = :ets.lookup(:tinkex_rate_limiters, key)
        existing
    end
  end

  def should_backoff?(limiter) do
    backoff_until = :atomics.get(limiter, 1)
    System.monotonic_time(:millisecond) < backoff_until
  end

  def set_backoff(limiter, duration_ms) do
    backoff_until = System.monotonic_time(:millisecond) + duration_ms
    :atomics.put(limiter, 1, backoff_until)
  end
end
```

#### Retry Behavior
- **TrainingClient**: Uses HTTP layer retries with exponential backoff
- **SamplingClient**: Sets max_retries: 0, surfaces errors immediately for caller to handle
- **429 Handling**: Uses server-provided retry_after_ms from response headers

### 3.6 Configuration Threading for Multi-Tenancy

Tinkex uses a Config struct for multi-tenant support:

```elixir
defmodule Tinkex.Config do
  defstruct [:base_url, :api_key, :http_pool, :timeout, :max_retries, :user_metadata]

  def new(opts \\ []) do
    %__MODULE__{
      base_url: opts[:base_url] || Application.get_env(:tinkex, :base_url),
      api_key: opts[:api_key] || Application.get_env(:tinkex, :api_key),
      http_pool: opts[:http_pool] || Tinkex.HTTP.Pool,
      timeout: opts[:timeout] || 120_000,
      max_retries: opts[:max_retries] || 3,
      user_metadata: opts[:user_metadata]
    }
  end
end
```

This enables multiple clients with different API keys or base URLs within the same BEAM instance.

---

## 4. Integration Points with Crucible

### 4.1 crucible_ensemble: Multi-Model Sampling

#### Integration Concept

Use Tinkex SamplingClients as model backends for ensemble voting across multiple LoRA adapters.

```elixir
defmodule Crucible.Ensemble.TinkexBackend do
  @behaviour Crucible.Ensemble.Backend

  @impl true
  def sample(client, prompt, opts) do
    sampling_params = build_sampling_params(opts)
    {:ok, task} = Tinkex.SamplingClient.sample(client, prompt, 1, sampling_params)

    case Task.await(task, opts[:timeout] || 30_000) do
      {:ok, response} ->
        {:ok, %{
          text: extract_text(response),
          confidence: extract_confidence(response),
          logprobs: response.logprobs
        }}
      {:error, _} = error ->
        error
    end
  end
end
```

#### Multi-Adapter Ensemble

```elixir
# Create ensemble from multiple LoRA adapters
adapters = [
  "claim-extractor-scifact-v1",
  "claim-extractor-scifact-v2",
  "claim-extractor-scifact-v3"
]

samplers = Enum.map(adapters, fn adapter_name ->
  {:ok, sampler} = Tinkex.ServiceClient.create_sampling_client(
    service,
    model_path: adapter_name,
    base_model: "meta-llama/Llama-3.1-8B-Instruct"
  )
  sampler
end)

# Run ensemble voting
result = Crucible.Ensemble.vote(
  samplers,
  prompt,
  strategy: :weighted,
  backend: Crucible.Ensemble.TinkexBackend
)
```

### 4.2 crucible_hedging: Leverage Tinkex Rate Limiting

#### Integration Concept

Use `crucible_hedging` strategies for tail latency reduction on Tinkex sampling operations.

```elixir
defmodule Crucible.Hedging.TinkexStrategy do
  @behaviour Crucible.Hedging.Strategy

  @impl true
  def execute(clients, request, opts) do
    # Implement hedged requests respecting Tinkex rate limits
    strategy = opts[:strategy] || :percentile

    case strategy do
      :fixed ->
        fixed_hedge(clients, request, opts)
      :percentile ->
        percentile_hedge(clients, request, opts)
      :adaptive ->
        adaptive_hedge(clients, request, opts)
    end
  end

  defp percentile_hedge(clients, request, opts) do
    # Start initial request
    primary_task = start_sample(hd(clients), request, opts)

    # Wait for P95 latency threshold, then start hedge
    case Task.yield(primary_task, opts[:p95_ms] || 500) do
      {:ok, result} ->
        result
      nil ->
        # Start hedge request to backup client
        hedge_task = start_sample(Enum.at(clients, 1), request, opts)

        # Return first to complete
        select_first_result([primary_task, hedge_task])
    end
  end
end
```

#### Rate Limiter Coordination

```elixir
# Check shared rate limiter before hedging
defp can_hedge?(client, config) do
  limiter = Tinkex.RateLimiter.for_key({config.base_url, config.api_key})
  not Tinkex.RateLimiter.should_backoff?(limiter)
end
```

### 4.3 crucible_telemetry: Connect Tinkex Events

#### Tinkex Telemetry Events

Tinkex emits standard telemetry events:

```elixir
# HTTP request events
[:tinkex, :http, :request, :start]
[:tinkex, :http, :request, :stop]
[:tinkex, :http, :request, :exception]

# Queue state changes
[:tinkex, :queue_state, :change]
```

#### Crucible Integration

```elixir
defmodule Crucible.Telemetry.TinkexHandler do
  def attach() do
    :telemetry.attach_many(
      "crucible-tinkex-handler",
      [
        [:tinkex, :http, :request, :stop],
        [:tinkex, :http, :request, :exception]
      ],
      &handle_event/4,
      %{experiment_id: nil}
    )
  end

  def handle_event([:tinkex, :http, :request, :stop], measurements, metadata, state) do
    event = %Crucible.Telemetry.Event{
      timestamp: DateTime.utc_now(),
      experiment_id: state.experiment_id,
      event_type: :tinkex_http_request,
      measurements: %{
        duration_ms: System.convert_time_unit(measurements.duration, :native, :millisecond),
        retry_count: metadata.retry_count
      },
      metadata: %{
        path: metadata.path,
        status: metadata.status,
        client_type: metadata.client_type
      }
    }

    Crucible.Telemetry.Research.capture(event)
  end
end
```

#### Experiment Isolation

```elixir
# Tag all Tinkex events with experiment context
Crucible.Telemetry.Research.with_experiment(experiment_id, fn ->
  {:ok, result} = Tinkex.TrainingClient.forward_backward(client, data, :cross_entropy)
  result
end)
```

### 4.4 crucible_bench: Statistical Testing of Fine-Tuned Models

#### Evaluation Workflow

```elixir
defmodule Crucible.Bench.TinkexEvaluator do
  @doc """
  Evaluate fine-tuned Tinkex models with statistical rigor.
  """
  def evaluate_adapter(adapter_name, test_data, opts \\ []) do
    # Create sampling client for adapter
    {:ok, sampler} = create_sampler(adapter_name, opts)

    # Run evaluation
    results = Enum.map(test_data, fn example ->
      {:ok, task} = Tinkex.SamplingClient.sample(sampler, example.prompt, 1, opts[:sampling_params])
      {:ok, response} = Task.await(task)

      %{
        expected: example.expected,
        actual: extract_text(response),
        metrics: compute_metrics(example, response)
      }
    end)

    # Statistical analysis
    Crucible.Bench.analyze(results, [
      tests: [:paired_t_test, :wilcoxon],
      effect_sizes: [:cohens_d],
      confidence: 0.95
    ])
  end
end
```

#### A/B Testing Adapters

```elixir
# Compare two adapter versions
results = Crucible.Bench.ab_test(
  adapter_a: "claim-extractor-v1",
  adapter_b: "claim-extractor-v2",
  test_data: scifact_dev,
  evaluator: Crucible.Bench.TinkexEvaluator,
  metrics: [:entailment_score, :citation_accuracy],
  tests: [:mann_whitney, :bootstrap_ci]
)

# Results include:
# - p-values for significance
# - Effect sizes (Cohen's d, Cliff's delta)
# - Confidence intervals
# - Power analysis
```

### 4.5 crucible_harness: Orchestrate Training Experiments

#### Experiment DSL

```elixir
defmodule MyExperiment do
  use Crucible.Harness.Experiment

  experiment "fine_tune_claim_extractor" do
    description "Fine-tune Llama-3.1-8B on SciFact with varying citation penalties"

    # Define hyperparameter sweep
    parameter :citation_validity_weight, values: [2.0, 3.0, 5.0, 7.0]
    parameter :learning_rate, values: [1.0e-4, 2.0e-4]
    parameter :lora_rank, values: [16, 32]

    # Training stage using Tinkex
    stage :train do
      backend Crucible.Harness.TinkexBackend

      config do
        base_model "meta-llama/Llama-3.1-8B-Instruct"
        dataset "scifact_claim_extractor"
        epochs 5
        batch_size 8
        lora_alpha param(:lora_rank) * 2
      end

      checkpoint_every 100
      telemetry :detailed
    end

    # Evaluation stage
    stage :evaluate do
      evaluator Crucible.Bench.TinkexEvaluator
      test_data "scifact_dev"

      metrics [
        :schema_compliance,
        :citation_accuracy,
        :mean_entailment,
        :overall_pass_rate
      ]
    end

    # Antagonist stage
    stage :antagonist do
      analyzer Crucible.Harness.AntagonistAnalyzer

      thresholds do
        chirality_trigger 0.55
        entailment_threshold 0.5
      end
    end

    # Statistical comparison
    stage :analyze do
      tests [:kruskal_wallis, :post_hoc_dunn]
      report_format [:markdown, :latex]
    end
  end
end

# Run experiment
Crucible.Harness.run(MyExperiment,
  output_dir: "./experiments/scifact_sweep",
  resume: true
)
```

#### Tinkex Backend Implementation

```elixir
defmodule Crucible.Harness.TinkexBackend do
  @behaviour Crucible.Harness.TrainingBackend

  @impl true
  def init(config) do
    tinkex_config = Tinkex.Config.new(
      api_key: config.api_key,
      base_url: config.base_url
    )

    {:ok, service} = Tinkex.ServiceClient.start_link(config: tinkex_config)

    {:ok, training_client} = Tinkex.ServiceClient.create_lora_training_client(
      service,
      base_model: config.base_model,
      rank: config.lora_rank,
      train_mlp: true,
      train_attn: true
    )

    {:ok, %{service: service, training_client: training_client, config: config}}
  end

  @impl true
  def train_step(state, batch, step) do
    data = prepare_data(batch, state.config)

    {:ok, task} = Tinkex.TrainingClient.forward_backward(
      state.training_client,
      data,
      :cross_entropy,
      loss_fn_config: %{
        citation_validity_weight: state.config.citation_validity_weight
      }
    )

    {:ok, result} = Task.await(task, :infinity)

    # Emit telemetry
    Crucible.Telemetry.Research.emit(:training_step, %{
      step: step,
      loss: result.total_loss,
      citation_invalid_rate: result.citation_invalid_rate
    })

    {:ok, result}
  end

  @impl true
  def optimize(state) do
    adam_params = %Tinkex.Types.AdamParams{
      learning_rate: state.config.learning_rate,
      beta1: 0.9,
      beta2: 0.999
    }

    {:ok, task} = Tinkex.TrainingClient.optim_step(state.training_client, adam_params)
    Task.await(task)
  end

  @impl true
  def checkpoint(state, name) do
    {:ok, sampler} = Tinkex.TrainingClient.save_weights_and_get_sampling_client(
      state.training_client,
      name: name
    )

    {:ok, %{sampler: sampler, adapter_name: name}}
  end
end
```

---

## 5. Proposed API Design

### 5.1 High-Level Training Abstractions

```elixir
defmodule Crucible.Tinkex do
  @moduledoc """
  High-level interface for Tinkex integration with Crucible.
  """

  @doc """
  Create a managed training session with Crucible telemetry and checkpointing.
  """
  def create_training_session(opts) do
    experiment_id = opts[:experiment_id] || Crucible.generate_id()

    # Initialize Tinkex
    config = Tinkex.Config.new(Keyword.take(opts, [:api_key, :base_url]))
    {:ok, service} = Tinkex.ServiceClient.start_link(config: config)

    {:ok, training_client} = Tinkex.ServiceClient.create_lora_training_client(
      service,
      Keyword.take(opts, [:base_model, :rank, :seed])
    )

    # Attach telemetry
    Crucible.Telemetry.TinkexHandler.attach(experiment_id)

    {:ok, %Crucible.Tinkex.Session{
      experiment_id: experiment_id,
      service: service,
      training_client: training_client,
      config: config
    }}
  end

  @doc """
  Run a complete training loop with automatic checkpointing and metrics.
  """
  def train(session, dataset, opts \\ []) do
    epochs = opts[:epochs] || 3
    batch_size = opts[:batch_size] || 8
    checkpoint_every = opts[:checkpoint_every] || 100

    batches = batch_dataset(dataset, batch_size)
    total_steps = length(batches) * epochs

    Enum.reduce(1..epochs, {:ok, []}, fn epoch, {:ok, metrics} ->
      epoch_metrics = Enum.with_index(batches, fn batch, idx ->
        step = (epoch - 1) * length(batches) + idx + 1

        # Forward-backward pass
        {:ok, fb_result} = forward_backward(session, batch, opts)

        # Optimization step
        {:ok, _} = optimize(session, opts)

        # Checkpoint if needed
        if rem(step, checkpoint_every) == 0 do
          checkpoint(session, "step_#{step}")
        end

        %{
          step: step,
          epoch: epoch,
          loss: fb_result.total_loss,
          citation_invalid_rate: fb_result.citation_invalid_rate
        }
      end)

      {:ok, metrics ++ epoch_metrics}
    end)
  end

  @doc """
  Create an ensemble from multiple adapters.
  """
  def create_ensemble(service, adapter_names, opts \\ []) do
    samplers = Enum.map(adapter_names, fn name ->
      {:ok, sampler} = Tinkex.ServiceClient.create_sampling_client(
        service,
        model_path: name,
        base_model: opts[:base_model]
      )
      sampler
    end)

    {:ok, %Crucible.Ensemble{
      samplers: samplers,
      strategy: opts[:strategy] || :majority,
      backend: Crucible.Ensemble.TinkexBackend
    }}
  end
end
```

### 5.2 Experiment Configuration DSL

```elixir
defmodule Crucible.Tinkex.ExperimentConfig do
  defmacro experiment(name, do: block) do
    quote do
      defmodule unquote(Module.concat([__MODULE__, Macro.camelize(name)])) do
        use Crucible.Harness.Experiment

        @name unquote(name)

        unquote(block)
      end
    end
  end

  defmacro training(opts) do
    quote do
      @training_config unquote(opts)
    end
  end

  defmacro evaluation(opts) do
    quote do
      @evaluation_config unquote(opts)
    end
  end

  defmacro quality_targets(opts) do
    quote do
      @quality_targets %{
        schema_compliance: unquote(opts[:schema_compliance] || 0.95),
        citation_accuracy: unquote(opts[:citation_accuracy] || 0.95),
        mean_entailment: unquote(opts[:mean_entailment] || 0.50),
        overall_pass_rate: unquote(opts[:overall_pass_rate] || 0.45)
      }
    end
  end
end
```

### 5.3 Result Aggregation and Reporting

```elixir
defmodule Crucible.Tinkex.Reporter do
  @moduledoc """
  Generate reports from Tinkex training experiments.
  """

  @doc """
  Generate comprehensive experiment report.
  """
  def generate_report(experiment_id, opts \\ []) do
    # Collect all telemetry data
    events = Crucible.Telemetry.Research.query(experiment_id)

    # Extract metrics
    training_metrics = extract_training_metrics(events)
    evaluation_metrics = extract_evaluation_metrics(events)

    # Statistical analysis
    analysis = Crucible.Bench.analyze(evaluation_metrics, opts[:tests] || [:all])

    # Generate report
    report = %Crucible.Report{
      experiment_id: experiment_id,
      title: opts[:title] || "Tinkex Training Experiment",
      sections: [
        %{name: "Training Progress", data: training_metrics},
        %{name: "Evaluation Results", data: evaluation_metrics},
        %{name: "Statistical Analysis", data: analysis},
        %{name: "Quality Targets", data: assess_targets(evaluation_metrics)}
      ],
      generated_at: DateTime.utc_now()
    }

    # Export in requested formats
    formats = opts[:formats] || [:markdown]

    Enum.map(formats, fn format ->
      path = Path.join(opts[:output_dir] || ".", "report.#{format}")
      Crucible.Reporter.export(report, format, path)
      path
    end)
  end

  defp assess_targets(metrics) do
    targets = %{
      schema_compliance: {0.95, metrics.schema_compliance},
      citation_accuracy: {0.95, metrics.citation_accuracy},
      mean_entailment: {0.50, metrics.mean_entailment},
      overall_pass_rate: {0.45, metrics.overall_pass_rate}
    }

    Enum.map(targets, fn {name, {target, actual}} ->
      %{
        metric: name,
        target: target,
        actual: actual,
        passed: actual >= target,
        delta: actual - target
      }
    end)
  end
end
```

---

## 6. Implementation Roadmap

### Phase 1: Core Tinkex Wrapper (Weeks 1-2)

#### Week 1: Foundation
- [ ] Create `crucible_tinkex` OTP application
- [ ] Implement `Crucible.Tinkex.Config` for Crucible-specific configuration
- [ ] Add `Crucible.Tinkex.Session` for managed training sessions
- [ ] Implement basic telemetry handler for Tinkex events
- [ ] Write unit tests with mocked Tinkex clients

#### Week 2: Training Abstractions
- [ ] Implement `Crucible.Tinkex.train/3` high-level training loop
- [ ] Add automatic checkpointing with Crucible naming conventions
- [ ] Implement loss function configuration with CNS3-style weights
- [ ] Add progress reporting and logging
- [ ] Write integration tests with actual Tinkex backend

#### Deliverables
- `crucible_tinkex` application with basic training support
- Telemetry integration for training events
- Documentation for basic usage

### Phase 2: Crucible Integration (Weeks 3-4)

#### Week 3: Ensemble and Hedging
- [ ] Implement `Crucible.Ensemble.TinkexBackend`
- [ ] Add support for multi-adapter ensembles
- [ ] Implement `Crucible.Hedging.TinkexStrategy`
- [ ] Add rate limiter coordination for hedged requests
- [ ] Write tests for concurrent sampling scenarios

#### Week 4: Evaluation and Reporting
- [ ] Implement `Crucible.Bench.TinkexEvaluator`
- [ ] Add A/B testing support for adapter comparison
- [ ] Integrate with `crucible_bench` statistical tests
- [ ] Implement `Crucible.Tinkex.Reporter` for result aggregation
- [ ] Add export formats (Markdown, LaTeX, Jupyter)

#### Deliverables
- Full ensemble and hedging support
- Evaluation framework with statistical analysis
- Report generation

### Phase 3: CNS Support (Weeks 5-6)

#### Week 5: Harness Backend
- [ ] Implement `Crucible.Harness.TinkexBackend`
- [ ] Add experiment DSL extensions for Tinkex
- [ ] Implement hyperparameter sweep support
- [ ] Add checkpoint resume capability
- [ ] Integrate Antagonist analysis stage

#### Week 6: Quality Targets and Polish
- [ ] Implement quality target assessment
- [ ] Add CNS3-specific metrics (entailment, chirality, citation)
- [ ] Write comprehensive documentation
- [ ] Create example experiments
- [ ] Performance optimization and testing

#### Deliverables
- Complete harness integration
- CNS3-compatible quality metrics
- Production-ready documentation
- Example experiments for reference

### Milestones

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| M1: Basic Training | Week 2 | Can train a model with Tinkex through Crucible API |
| M2: Ensemble Support | Week 3 | Can run ensemble voting with multiple Tinkex adapters |
| M3: Statistical Evaluation | Week 4 | Can A/B test adapters with significance testing |
| M4: Harness Integration | Week 5 | Can run complete experiment from DSL |
| M5: Production Ready | Week 6 | All tests passing, documentation complete |

---

## 7. Quality Targets from Thinker Experiments

### 7.1 Primary Quality Metrics

Based on CNS3 experiments, the following quality targets must be met for any integration to be considered successful:

| Metric | Target | Description | Measurement |
|--------|--------|-------------|-------------|
| Schema Compliance | >= 95% | Outputs must conform to expected structure | JSON schema validation |
| Citation Accuracy | >= 95% | Only valid citations from source corpus | Citation validation against corpus |
| Mean Entailment | >= 0.50 | Claims must be supported by evidence | NLI model entailment score |
| Overall Pass Rate | >= 45% | Combined semantic quality threshold | (entailment >= 0.75) AND (similarity >= 0.70) |

### 7.2 Secondary Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Entailment Pass Rate | >= 45% | Individual claims with entailment >= 0.75 |
| Similarity Pass Rate | >= 35% | Individual claims with similarity >= 0.70 |
| HIGH Severity Flags | 0 | No CITATION_INVALID or critical issues |
| Chirality Score | < 0.55 mean | Low polarity contradiction |

### 7.3 Training Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Loss Reduction | >= 95% | Training loss reduction from start to end |
| Citation Invalid Rate | 0.000 | No invalid citations in training data |
| Convergence | < 500 steps | Loss should stabilize within step limit |

### 7.4 Validation Workflow

```elixir
defmodule Crucible.Tinkex.QualityValidator do
  @targets %{
    schema_compliance: 0.95,
    citation_accuracy: 0.95,
    mean_entailment: 0.50,
    overall_pass_rate: 0.45
  }

  def validate(evaluation_results) do
    assessments = Enum.map(@targets, fn {metric, target} ->
      actual = Map.get(evaluation_results, metric, 0)
      passed = actual >= target

      %{
        metric: metric,
        target: target,
        actual: actual,
        passed: passed,
        delta: Float.round(actual - target, 3)
      }
    end)

    all_passed = Enum.all?(assessments, & &1.passed)

    %{
      passed: all_passed,
      assessments: assessments,
      summary: summarize(assessments)
    }
  end

  defp summarize(assessments) do
    passed_count = Enum.count(assessments, & &1.passed)
    total_count = length(assessments)

    "#{passed_count}/#{total_count} quality targets met"
  end
end
```

### 7.5 Continuous Quality Monitoring

The integration should support continuous monitoring during experiments:

```elixir
# In experiment configuration
quality_monitoring do
  check_every :epoch

  thresholds do
    warn_if :mean_entailment, below: 0.40
    fail_if :citation_accuracy, below: 0.90
    fail_if :schema_compliance, below: 0.90
  end

  on_warning fn metrics ->
    Logger.warn("Quality degradation detected: #{inspect(metrics)}")
  end

  on_failure fn metrics ->
    Logger.error("Quality target failed: #{inspect(metrics)}")
    :stop_experiment
  end
end
```

---

## 8. Appendices

### A. Dependencies

```elixir
# In mix.exs
defp deps do
  [
    {:tinkex, "~> 0.1.0"},
    {:crucible_framework, path: "../crucible_framework"},
    {:crucible_bench, path: "../crucible_bench"},
    {:crucible_telemetry, path: "../crucible_telemetry"},
    {:crucible_ensemble, path: "../crucible_ensemble"},
    {:crucible_hedging, path: "../crucible_hedging"},
    {:crucible_harness, path: "../crucible_harness"}
  ]
end
```

### B. Configuration Example

```elixir
# config/config.exs
import Config

config :crucible_framework,
  default_base_model: "meta-llama/Llama-3.1-8B-Instruct",
  default_lora_rank: 16,
  quality_targets: %{
    schema_compliance: 0.95,
    citation_accuracy: 0.95,
    mean_entailment: 0.50,
    overall_pass_rate: 0.45
  }

config :tinkex,
  api_key: System.get_env("TINKER_API_KEY"),
  base_url: "https://tinker.thinkingmachines.dev/services/tinker-prod"
```

### C. Example Experiment

```elixir
defmodule Examples.SciFactExperiment do
  use Crucible.Harness.Experiment

  experiment "scifact_claim_extractor_sweep" do
    description """
    Fine-tune Llama-3.1-8B on SciFact claim extraction task with
    varying citation validity penalties. Based on CNS3 experiments.
    """

    parameter :citation_validity_weight, values: [2.0, 5.0, 7.0]
    parameter :learning_rate, values: [2.0e-4]
    parameter :lora_rank, values: [16]

    stage :train do
      backend Crucible.Harness.TinkexBackend

      config do
        base_model "meta-llama/Llama-3.1-8B-Instruct"
        dataset "scifact_claim_extractor"
        epochs 5
        batch_size 8
        lora_alpha 32
      end

      checkpoint_every 100
    end

    stage :evaluate do
      evaluator Crucible.Bench.TinkexEvaluator
      test_data "scifact_dev"
      max_samples 50

      metrics [
        :schema_compliance,
        :citation_accuracy,
        :mean_entailment,
        :entailment_pass_rate,
        :similarity_pass_rate,
        :overall_pass_rate
      ]
    end

    stage :antagonist do
      analyzer Crucible.Harness.AntagonistAnalyzer

      thresholds do
        chirality_trigger 0.55
        entailment_threshold 0.5
      end

      output "antagonist_flags.jsonl"
    end

    stage :analyze do
      tests [:kruskal_wallis, :post_hoc_dunn]

      compare_by :citation_validity_weight

      report_format [:markdown, :latex]
      output_dir "./results/scifact_sweep"
    end
  end
end
```

### D. Glossary

| Term | Definition |
|------|------------|
| **LoRA** | Low-Rank Adaptation - efficient fine-tuning technique |
| **CNS** | CNS Support Models - the experimental ML project |
| **Antagonist** | Quality evaluation component that flags issues |
| **Chirality** | Polarity contradiction metric in claim graphs |
| **Entailment** | Degree to which claims are supported by evidence |
| **SamplingClient** | Tinkex client for text generation |
| **TrainingClient** | Tinkex client for model fine-tuning |
| **Harness** | Crucible experiment orchestration system |

---

## 9. References

1. CNS3 Antagonist MVP RFC (2025-11-18)
2. Thinker Specification - Transformers + PEFT + TDD Orchestrator
3. Training Results - Citation Validation (2025-11-18)
4. Tinkex Client Architecture Analysis
5. Crucible Framework README

---

**Document Status:** Draft
**Next Review:** 2025-11-28
**Maintainers:** North-Shore-AI Team
