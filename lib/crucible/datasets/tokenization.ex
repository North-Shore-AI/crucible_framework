defmodule Crucible.Datasets.Tokenization do
  @moduledoc """
  Tokenizer integration for Tinkex training data.

  Provides utilities for:
  - Text tokenization and encoding
  - Batch padding and truncation
  - Converting to Tinkex datum types
  - Special token handling
  """

  alias Tinkex.Types.{Datum, ModelInput, TensorData, ForwardBackwardInput}

  @type tokenizer :: :tinkex | :tiktoken | :huggingface | :custom

  # Default special tokens (typical values for LLaMA-style models)
  @bos_token 1
  @eos_token 2
  @pad_token 0

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Tokenizes text into a list of token strings.

  ## Options
  - `:tokenizer` - Tokenizer to use. Default: :tinkex
  """
  @spec tokenize(String.t(), keyword()) :: [String.t()]
  def tokenize(text, opts \\ []) do
    _tokenizer = Keyword.get(opts, :tokenizer, :tinkex)

    # Simple whitespace tokenization for demonstration
    # In production, would use actual tokenizer
    text
    |> String.split(~r/\s+/, trim: true)
    |> Enum.flat_map(fn word ->
      # Simple subword tokenization simulation
      if String.length(word) > 4 do
        [String.slice(word, 0, 4), String.slice(word, 4, 100)]
        |> Enum.filter(&(&1 != ""))
      else
        [word]
      end
    end)
  end

  @doc """
  Detokenizes tokens back to text.
  """
  @spec detokenize([String.t()], keyword()) :: String.t()
  def detokenize(tokens, _opts \\ []) do
    Enum.join(tokens, " ")
  end

  @doc """
  Encodes text to token IDs.

  ## Options
  - `:tokenizer` - Tokenizer to use. Default: :tinkex
  - `:add_bos` - Add beginning of sequence token. Default: false
  - `:add_eos` - Add end of sequence token. Default: false
  """
  @spec encode(String.t(), keyword()) :: [integer()]
  def encode(text, opts \\ []) do
    add_bos = Keyword.get(opts, :add_bos, false)
    add_eos = Keyword.get(opts, :add_eos, false)

    if text == "" do
      []
    else
      tokens = tokenize(text, opts)

      # Simple hash-based encoding for demonstration
      ids =
        Enum.map(tokens, fn token ->
          # Offset to avoid special tokens
          :erlang.phash2(token, 32000) + 100
        end)

      ids = if add_bos, do: [@bos_token | ids], else: ids
      ids = if add_eos, do: ids ++ [@eos_token], else: ids

      ids
    end
  end

  @doc """
  Decodes token IDs back to text.
  """
  @spec decode([integer()], keyword()) :: String.t()
  def decode(ids, _opts \\ []) do
    # Filter out special tokens
    ids
    |> Enum.reject(&(&1 in [@bos_token, @eos_token, @pad_token]))
    |> Enum.map(fn id -> "token_#{id}" end)
    |> Enum.join(" ")
  end

  @doc """
  Pads a batch of token sequences to the same length.

  ## Options
  - `:max_length` - Maximum sequence length. Default: max in batch
  - `:pad_token` - Token ID for padding. Default: 0
  - `:padding_side` - :left or :right. Default: :right
  """
  @spec pad_batch([[integer()]], keyword()) :: [[integer()]]
  def pad_batch(batch, opts \\ []) do
    pad_token = Keyword.get(opts, :pad_token, @pad_token)
    padding_side = Keyword.get(opts, :padding_side, :right)

    max_len_in_batch = batch |> Enum.map(&length/1) |> Enum.max(fn -> 0 end)
    max_length = Keyword.get(opts, :max_length, max_len_in_batch)

    Enum.map(batch, fn seq ->
      seq = truncate(seq, max_length)
      pad_length = max_length - length(seq)
      padding = List.duplicate(pad_token, pad_length)

      case padding_side do
        :left -> padding ++ seq
        :right -> seq ++ padding
      end
    end)
  end

  @doc """
  Truncates a token sequence to a maximum length.

  ## Options
  - `:side` - :left or :right truncation. Default: :right
  """
  @spec truncate([integer()], pos_integer(), keyword()) :: [integer()]
  def truncate(tokens, max_length, opts \\ []) do
    side = Keyword.get(opts, :side, :right)

    if length(tokens) <= max_length do
      tokens
    else
      case side do
        :right -> Enum.take(tokens, max_length)
        :left -> Enum.take(tokens, -max_length)
      end
    end
  end

  @doc """
  Converts text to a Tinkex Datum structure.

  ## Options
  - `:add_bos` - Add beginning of sequence token
  - `:add_eos` - Add end of sequence token
  - `:targets` - Target text for loss computation
  """
  @spec to_datum(String.t(), keyword()) :: Datum.t()
  def to_datum(text, opts \\ []) do
    input_ids = encode(text, opts)
    targets = Keyword.get(opts, :targets)

    # Use ModelInput.from_ints to create proper structure with EncodedTextChunk
    model_input = ModelInput.from_ints(input_ids)

    loss_fn_inputs =
      if targets do
        target_ids = encode(targets, opts)

        %{
          "targets" => %TensorData{
            data: target_ids,
            dtype: :int64,
            shape: [length(target_ids)]
          }
        }
      else
        %{}
      end

    Datum.new(%{
      model_input: model_input,
      loss_fn_inputs: loss_fn_inputs
    })
  end

  @doc """
  Converts a batch of examples to ForwardBackwardInput.

  ## Options
  - `:loss_fn` - Loss function to use. Default: :cross_entropy
  - `:loss_fn_config` - Configuration for loss function
  - `:max_length` - Maximum sequence length for padding
  """
  @spec to_forward_backward_input([map()], keyword()) :: ForwardBackwardInput.t()
  def to_forward_backward_input(batch, opts \\ []) do
    loss_fn = Keyword.get(opts, :loss_fn, :cross_entropy)
    loss_fn_config = Keyword.get(opts, :loss_fn_config)

    data =
      Enum.map(batch, fn example ->
        input_text = example[:input] || example["input"]
        output_text = example[:output] || example["output"]

        to_datum(input_text, Keyword.put(opts, :targets, output_text))
      end)

    %ForwardBackwardInput{
      data: data,
      loss_fn: loss_fn,
      loss_fn_config: loss_fn_config
    }
  end

  # ============================================================================
  # Special Tokens
  # ============================================================================

  @doc """
  Returns the beginning of sequence token ID.
  """
  @spec bos_token() :: integer()
  def bos_token, do: @bos_token

  @doc """
  Returns the end of sequence token ID.
  """
  @spec eos_token() :: integer()
  def eos_token, do: @eos_token

  @doc """
  Returns the padding token ID.
  """
  @spec pad_token() :: integer()
  def pad_token, do: @pad_token
end
