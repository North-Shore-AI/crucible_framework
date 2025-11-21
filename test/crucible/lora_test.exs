defmodule Crucible.LoraTest do
  use ExUnit.Case, async: true

  defmodule FakeAdapter do
    @behaviour Crucible.Lora.Adapter

    def generate_id, do: "fake-id"

    def create_experiment(opts), do: {:ok, %{backend: :fake, opts: opts}}

    def batch_dataset(dataset, batch_size), do: Enum.chunk_every(dataset, batch_size)

    def format_training_data(batch, opts) do
      Enum.map(batch, &Map.put(&1, :formatted?, opts[:flag] || false))
    end

    def calculate_metrics(results), do: %{results: results, adapter: :fake}

    def validate_quality(results, _config), do: %{validated: results}

    def sampling_params(opts), do: Map.new(opts)

    def checkpoint_name(experiment_id, step), do: "#{experiment_id}-#{step}-fake"
  end

  setup do
    original = Application.get_env(:crucible_framework, :lora_adapter)
    Application.put_env(:crucible_framework, :lora_adapter, FakeAdapter)

    on_exit(fn ->
      if is_nil(original) do
        Application.delete_env(:crucible_framework, :lora_adapter)
      else
        Application.put_env(:crucible_framework, :lora_adapter, original)
      end
    end)

    :ok
  end

  test "delegates generate_id to adapter" do
    assert Crucible.Lora.generate_id() == "fake-id"
  end

  test "delegates experiment creation" do
    assert {:ok, %{backend: :fake}} = Crucible.Lora.create_experiment(name: "test")
  end

  test "delegates batch utilities" do
    assert Crucible.Lora.batch_dataset([1, 2, 3], 2) == [[1, 2], [3]]
  end

  test "delegates formatting" do
    batch = [%{input: "a"}]
    assert [%{formatted?: true}] = Crucible.Lora.format_training_data(batch, flag: true)
  end

  test "delegates metrics helpers" do
    assert %{adapter: :fake} = Crucible.Lora.calculate_metrics([%{loss: 1.0}])
  end

  test "delegates quality validation" do
    assert %{validated: %{}} = Crucible.Lora.validate_quality(%{}, %{})
  end

  test "delegates sampling params" do
    assert %{temperature: 0.5} = Crucible.Lora.sampling_params(temperature: 0.5)
  end

  test "delegates checkpoint naming" do
    assert Crucible.Lora.checkpoint_name("exp", 10) == "exp-10-fake"
  end
end
