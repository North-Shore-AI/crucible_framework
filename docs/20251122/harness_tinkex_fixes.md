# Harness Tinkex Integration Fixes

## Current Problem

The harness is passing raw text maps to Tinkex:
```elixir
%{input: "...", output: "...", weight: 1.0}
```

But `Tinkex.TrainingClient.forward_backward` expects Datum objects:
```elixir
%{
  model_input: %Tinkex.Types.ModelInput{chunks: [%EncodedTextChunk{tokens: [...]}]},
  loss_fn_inputs: %{
    "target_tokens" => %{data: [...], dtype: "int64", shape: [...]},
    "weights" => %{data: [...], dtype: "float32", shape: [...]}
  }
}
```

## What Python Does (train_claim_extractor.py)

### 1. Create clients
```python
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model=model_cfg["base_model"])
tokenizer = training_client.get_tokenizer()
```

### 2. Build Datums with tokenizer
```python
def build_datum(example, tokenizer, citation_penalty_multiplier=1.0):
    prompt_tokens = tokenizer.encode(example.prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(" " + example.completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    # Build weights array (same length as target_tokens)
    weights = [...]  # calculated per token

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=tinker.TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            weights=tinker.TensorData(data=weights, dtype="float32", shape=[len(weights)])
        )
    )
```

### 3. Call forward_backward
```python
datums = [build_datum(ex, tokenizer) for ex in batch]
fwdbwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
result = fwdbwd_future.result()
```

## Required Changes

### File: `lib/crucible/thinker/harness.ex`

#### 1. Remove Crucible.Tinkex.Session usage

Current (wrong):
```elixir
{:ok, session} = Tinkex.start_session(tinkex_exp)
```

Change to:
```elixir
{:ok, service} = Tinkex.ServiceClient.start_link(config: build_tinkex_config(experiment))
{:ok, training_client} = Tinkex.ServiceClient.create_lora_training_client(service,
  base_model: experiment.training_config.base_model
)
```

#### 2. Add build_datum function

```elixir
defp build_datum(sample, model_name, opts \\ []) do
  prompt = sample.input
  completion = " " <> sample.output

  # Tokenize
  {:ok, prompt_tokens} = Tinkex.Tokenizer.encode(prompt, model_name, opts)
  {:ok, completion_tokens} = Tinkex.Tokenizer.encode(completion, model_name, opts)

  tokens = prompt_tokens ++ completion_tokens
  input_tokens = Enum.slice(tokens, 0..-2//1)
  target_tokens = Enum.slice(tokens, 1..-1//1)

  # Build weights (all 1.0 for now, can add citation weighting later)
  weights = List.duplicate(1.0, length(target_tokens))

  # Create Datum
  Tinkex.Types.Datum.new(%{
    model_input: Tinkex.Types.ModelInput.from_ints(input_tokens),
    loss_fn_inputs: %{
      target_tokens: target_tokens,
      weights: weights
    }
  })
end
```

#### 3. Update run_training to use TrainingClient directly

Current (wrong):
```elixir
training_batch = Tinkex.format_training_data(batch, ...)  # This function doesn't exist!
{:ok, fb_result} = Tinkex.forward_backward(session, training_batch)
```

Change to:
```elixir
model_name = experiment.training_config.base_model
datums = Enum.map(batch, &build_datum(&1, model_name))

{:ok, task} = Tinkex.TrainingClient.forward_backward(training_client, datums, :cross_entropy)
fb_result = Task.await(task, :infinity)
```

#### 4. Update optim_step call

Current:
```elixir
{:ok, _} = Tinkex.optim_step(session, %{lr: ..., beta1: ..., ...})
```

Change to:
```elixir
adam_params = %{
  learning_rate: lora_config.learning_rate,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1.0e-8,
  weight_decay: lora_config.weight_decay
}
{:ok, task} = Tinkex.TrainingClient.optim_step(training_client, adam_params)
Task.await(task, :infinity)
```

### File: `lib/crucible/thinker/datasets/scifact.ex`

No changes needed - already returns `%{input: ..., output: ...}` which is correct input to `build_datum`.

### File: `lib/crucible/tinkex/session.ex`

This file can be removed or deprecated - it's a wrapper that's not needed. Use Tinkex directly.

## Summary of Changes

| What | Action |
|------|--------|
| `Tinkex.format_training_data` | Remove - doesn't exist |
| `Tinkex.forward_backward(session, ...)` | Replace with `Tinkex.TrainingClient.forward_backward(client, datums, loss_fn)` |
| `Tinkex.optim_step(session, ...)` | Replace with `Tinkex.TrainingClient.optim_step(client, adam_params)` |
| `Tinkex.start_session` | Replace with `Tinkex.ServiceClient.start_link` + `create_lora_training_client` |
| `Crucible.Tinkex.Session` | Don't use - go directly to Tinkex |
| Raw `%{input:, output:}` maps | Convert to Datums via `build_datum` with tokenization |

## Testing

After changes, run:
```bash
mix thinker
# Select option 4 (limited training)
```

Should see training progress without the `estimate_number_count` error.
