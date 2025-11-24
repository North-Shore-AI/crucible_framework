defmodule Crucible.Backend.Tinkex.LiveClient do
  @moduledoc """
  Default client implementation that delegates to the Tinkex SDK.
  """

  @behaviour Crucible.Backend.Tinkex.Client

  alias Tinkex.{SamplingClient, ServiceClient, TrainingClient}
  alias Tinkex.Types.{AdamParams, SamplingParams}

  @impl true
  def start_service(config) do
    apply(ServiceClient, :start_link, [[config: config]])
  end

  @impl true
  def create_training_client(service, opts) do
    apply(ServiceClient, :create_lora_training_client, [service, opts])
  end

  @impl true
  def forward_backward(training_client, data, loss_fn, opts) do
    apply(TrainingClient, :forward_backward, [training_client, data, loss_fn, opts])
  end

  @impl true
  def optim_step(training_client, params) do
    adam_params = struct(AdamParams, params)
    apply(TrainingClient, :optim_step, [training_client, adam_params])
  end

  @impl true
  def save_weights_and_get_sampler(training_client) do
    apply(TrainingClient, :save_weights_and_get_sampling_client, [training_client])
  end

  @impl true
  def sample(sampler, prompt, params, opts) do
    sampling_params = struct(SamplingParams, params)
    num_samples = Map.get(opts, :num_samples, 1)

    apply(SamplingClient, :sample, [
      sampler,
      prompt: prompt,
      sampling_params: sampling_params,
      num_samples: num_samples
    ])
  end
end
