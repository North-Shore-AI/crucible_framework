defmodule Crucible.Backend.Tinkex do
  @moduledoc """
  `Crucible.Backend` implementation backed by the Tinkex SDK.

  The implementation is thin and delegates network calls to a configurable
  client module so tests can use mocks while live runs hit the real API.
  """

  @behaviour Crucible.Backend

  require Logger

  alias Crucible.Backend.Tinkex.LiveClient
  alias Tinkex.Config
  alias Tinkex.Types.{Datum, ModelInput, TensorData}

  @impl true
  def init(_backend_id, config_opts) when is_map(config_opts) do
    env_api_key = System.get_env("TINKER_API_KEY") || Application.get_env(:tinkex, :api_key)
    env_base_url = System.get_env("TINKER_BASE_URL") || Application.get_env(:tinkex, :base_url)

    config =
      config_opts
      |> Map.update(:api_key, env_api_key, fn current -> current || env_api_key end)
      |> Map.update(:base_url, env_base_url, fn current -> current || env_base_url end)
      |> Map.to_list()
      |> Config.new()

    {:ok,
     %{
       config: config,
       client_mod: client_mod()
     }}
  rescue
    e ->
      {:error, {:invalid_config, Exception.message(e)}}
  end

  @impl true
  def start_session(%{config: config, client_mod: client_mod} = state, experiment) do
    training_opts =
      experiment.backend.options
      |> Map.new()
      |> Map.put_new(
        :base_model,
        experiment.backend.options[:base_model] || "meta-llama/Llama-3.2-1B"
      )
      |> Keyword.new()

    with {:ok, service} <- client_mod.start_service(config),
         {:ok, training_client} <- client_mod.create_training_client(service, training_opts) do
      {:ok,
       %{
         service: service,
         training_client: training_client,
         config: config,
         client_mod: client_mod,
         experiment: experiment,
         state: state
       }}
    end
  end

  @impl true
  def train_step(
        %{training_client: training_client, experiment: experiment, client_mod: client_mod},
        batch
      ) do
    with {:ok, data} <- encode_batch(batch, experiment),
         {:ok, result} <-
           forward_backward(
             client_mod,
             training_client,
             data,
             loss_fn(experiment),
             timeout(experiment)
           ) do
      metrics = metrics_from(result)
      safe_metrics = sanitize_result(result)

      {:ok,
       %{
         loss: loss_from(result, metrics),
         batch_size: batch_size(result, batch),
         metrics: safe_metrics
       }}
    end
  end

  defp loss_fn(experiment), do: Map.get(experiment.backend.options, :loss_fn, :cross_entropy)

  defp timeout(experiment), do: Map.get(experiment.backend.options, :train_timeout, 30_000)

  defp forward_backward(client_mod, training_client, data, loss_fn, timeout) do
    case client_mod.forward_backward(training_client, data, loss_fn, []) do
      {:ok, %Task{} = task} ->
        case Task.await(task, timeout) do
          {:ok, result} -> {:ok, result}
          {:error, reason} -> {:error, reason}
        end

      {:ok, %{} = result} ->
        {:ok, result}

      {:error, reason} ->
        {:error, reason}

      other ->
        {:error, {:unexpected_forward_backward_result, other}}
    end
  rescue
    e ->
      {:error, {:forward_backward_failed, Exception.message(e)}}
  end

  defp loss_from(result, metrics) do
    cond do
      is_map(metrics) && Map.get(metrics, "loss") -> Map.get(metrics, "loss")
      is_map(metrics) && Map.get(metrics, :loss) -> Map.get(metrics, :loss)
      Map.get(result, :total_loss) -> Map.get(result, :total_loss)
      Map.get(result, "total_loss") -> Map.get(result, "total_loss")
      true -> 0.0
    end
  end

  defp batch_size(result, batch) do
    Map.get(result, :num_examples) || Map.get(result, "num_examples") || length(batch)
  end

  defp metrics_from(%{} = result) do
    cond do
      Map.has_key?(result, :metrics) -> Map.get(result, :metrics) || %{}
      Map.has_key?(result, "metrics") -> Map.get(result, "metrics") || %{}
      true -> %{}
    end
  end

  defp metrics_from(_), do: %{}

  defp sanitize_result(%_{} = struct) do
    struct
    |> Map.from_struct()
    |> sanitize_result()
  end

  defp sanitize_result(map) when is_map(map) do
    map
    |> Enum.map(fn {k, v} -> {sanitize_key(k), sanitize_result(v)} end)
    |> Enum.into(%{})
  end

  defp sanitize_result(list) when is_list(list), do: Enum.map(list, &sanitize_result/1)

  defp sanitize_result(tuple) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> sanitize_result()
  end

  defp sanitize_result(value)
       when is_pid(value) or is_reference(value) or is_function(value) or is_port(value),
       do: inspect(value)

  defp sanitize_result(value), do: value

  defp sanitize_key(key) when is_atom(key), do: Atom.to_string(key)
  defp sanitize_key(key), do: key

  @impl true
  def save_checkpoint(%{training_client: training_client, client_mod: client_mod}, _step) do
    case client_mod.save_weights_and_get_sampler(training_client) do
      {:ok, sampler} -> {:ok, sampler}
      other -> other
    end
  end

  @impl true
  def create_sampler(_session, sampler) when not is_nil(sampler), do: {:ok, sampler}
  def create_sampler(_session, _checkpoint_ref), do: {:error, :no_sampler_available}

  @impl true
  def sample(sampler, prompt, opts) do
    model_name = Map.get(opts, :model_name, "meta-llama/Llama-3.2-1B")

    with {:ok, prompt_input} <- ModelInput.from_text(prompt, model_name: model_name) do
      params = %{
        temperature: Map.get(opts, :temperature, 0.7),
        top_p: Map.get(opts, :top_p, 0.95),
        max_tokens: Map.get(opts, :max_tokens, 128)
      }

      num_samples = Map.get(opts, :num_samples, 1)

      case client_mod().sample(sampler, prompt_input, params, %{num_samples: num_samples}) do
        {:ok, %Task{} = task} ->
          case Task.await(task, Map.get(opts, :timeout, 15_000)) do
            {:ok, responses} -> {:ok, unwrap_samples(responses)}
            {:error, reason} -> {:error, reason}
          end

        {:ok, responses} ->
          {:ok, unwrap_samples(responses)}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp encode_batch(batch, experiment) do
    model_name = experiment.backend.options[:base_model] || "meta-llama/Llama-3.2-1B"

    data =
      batch
      |> Enum.map(&encode_example(&1, model_name))
      |> Enum.reject(&is_nil/1)

    case data do
      [] -> {:error, :empty_batch}
      _ -> {:ok, data}
    end
  end

  defp encode_example(%Datum{} = datum, _model_name), do: datum

  defp encode_example(%{model_input: _, loss_fn_inputs: _} = datum, _model_name) do
    struct(Datum, datum)
  end

  defp encode_example(%{input: input, output: output}, model_name) do
    with {:ok, prompt_input} <-
           ModelInput.from_text("#{input}\n#{output}", model_name: model_name),
         {:ok, target_input} <- ModelInput.from_text(output, model_name: model_name) do
      tokens = ModelInput.to_ints(target_input)
      weights = List.duplicate(1.0, length(tokens))

      %Datum{
        model_input: prompt_input,
        loss_fn_inputs: %{
          "target_tokens" => %TensorData{data: tokens, dtype: :int64, shape: [length(tokens)]},
          "weights" => %TensorData{data: weights, dtype: :float32, shape: [length(tokens)]}
        }
      }
    else
      _ -> nil
    end
  end

  defp unwrap_samples(responses) when is_list(responses) do
    Enum.map(responses, fn resp ->
      cond do
        is_binary(resp) -> resp
        is_map(resp) -> Map.get(resp, :text) || Map.get(resp, "text") || inspect(resp)
        true -> inspect(resp)
      end
    end)
  end

  defp client_mod, do: Application.get_env(:crucible_framework, :tinkex_client, LiveClient)
end
