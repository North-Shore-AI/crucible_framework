defmodule Crucible.Datasets.TokenizationTest do
  use ExUnit.Case, async: true

  alias Crucible.Datasets.Tokenization

  describe "tokenize/2" do
    test "tokenizes text into tokens" do
      tokens = Tokenization.tokenize("Hello world")

      assert is_list(tokens)
      assert length(tokens) > 0
    end

    test "respects tokenizer option" do
      tokens = Tokenization.tokenize("Hello", tokenizer: :tinkex)

      assert is_list(tokens)
    end
  end

  describe "encode/2" do
    test "encodes text to token IDs" do
      ids = Tokenization.encode("Hello world")

      assert is_list(ids)
      assert Enum.all?(ids, &is_integer/1)
    end

    test "handles empty text" do
      ids = Tokenization.encode("")

      assert ids == []
    end
  end

  describe "decode/2" do
    test "decodes token IDs back to text" do
      text = "Hello world"
      ids = Tokenization.encode(text)
      decoded = Tokenization.decode(ids)

      assert is_binary(decoded)
    end
  end

  describe "to_datum/2" do
    test "creates Tinkex datum" do
      datum = Tokenization.to_datum("Test input")

      assert %Tinkex.Types.Datum{} = datum
      assert datum.model_input != nil
    end

    test "handles special tokens" do
      datum = Tokenization.to_datum("Test", add_bos: true, add_eos: true)

      assert %Tinkex.Types.Datum{} = datum
    end

    test "includes loss_fn_inputs when targets provided" do
      datum = Tokenization.to_datum("Input", targets: "Output")

      assert %Tinkex.Types.Datum{} = datum
      assert map_size(datum.loss_fn_inputs) > 0
    end
  end

  describe "to_forward_backward_input/2" do
    test "creates ForwardBackwardInput from batch" do
      batch = [
        %{input: "Example 1", output: "Label 1"},
        %{input: "Example 2", output: "Label 2"}
      ]

      fb_input = Tokenization.to_forward_backward_input(batch)

      assert %Tinkex.Types.ForwardBackwardInput{} = fb_input
      assert is_list(fb_input.data)
      assert length(fb_input.data) == 2
    end

    test "respects loss_fn option" do
      batch = [%{input: "Test", output: "Label"}]

      fb_input = Tokenization.to_forward_backward_input(batch, loss_fn: :cross_entropy)

      assert fb_input.loss_fn == :cross_entropy
    end

    test "handles loss_fn_config" do
      batch = [%{input: "Test", output: "Label"}]
      config = %{reduction: :mean}

      fb_input =
        Tokenization.to_forward_backward_input(batch,
          loss_fn: :cross_entropy,
          loss_fn_config: config
        )

      assert fb_input.loss_fn_config == config
    end
  end

  describe "pad_batch/2" do
    test "pads to max length in batch" do
      batch = [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1, 2]
      ]

      padded = Tokenization.pad_batch(batch)

      assert Enum.all?(padded, fn seq -> length(seq) == 5 end)
    end

    test "uses pad token" do
      batch = [
        [1, 2, 3],
        [1, 2]
      ]

      padded = Tokenization.pad_batch(batch, pad_token: 0)

      assert Enum.at(padded, 1) == [1, 2, 0]
    end

    test "respects max_length option" do
      batch = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2]
      ]

      padded = Tokenization.pad_batch(batch, max_length: 5)

      assert Enum.all?(padded, fn seq -> length(seq) == 5 end)
    end

    test "handles padding side" do
      batch = [[1, 2], [1, 2, 3]]

      left_padded = Tokenization.pad_batch(batch, padding_side: :left, pad_token: 0)
      right_padded = Tokenization.pad_batch(batch, padding_side: :right, pad_token: 0)

      assert hd(left_padded) == [0, 1, 2]
      assert hd(right_padded) == [1, 2, 0]
    end
  end

  describe "truncate/2" do
    test "truncates to max length" do
      tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

      truncated = Tokenization.truncate(tokens, 5)

      assert length(truncated) == 5
      assert truncated == [1, 2, 3, 4, 5]
    end

    test "returns unchanged if under max length" do
      tokens = [1, 2, 3]

      truncated = Tokenization.truncate(tokens, 10)

      assert truncated == tokens
    end

    test "handles truncation side" do
      tokens = [1, 2, 3, 4, 5]

      left = Tokenization.truncate(tokens, 3, side: :left)
      right = Tokenization.truncate(tokens, 3, side: :right)

      assert left == [3, 4, 5]
      assert right == [1, 2, 3]
    end
  end

  describe "special tokens" do
    test "bos_token returns beginning of sequence token" do
      token = Tokenization.bos_token()

      assert is_integer(token) or is_binary(token)
    end

    test "eos_token returns end of sequence token" do
      token = Tokenization.eos_token()

      assert is_integer(token) or is_binary(token)
    end

    test "pad_token returns padding token" do
      token = Tokenization.pad_token()

      assert is_integer(token) or is_binary(token)
    end
  end

  describe "detokenize/2" do
    test "converts tokens back to text" do
      tokens = Tokenization.tokenize("Hello world")
      text = Tokenization.detokenize(tokens)

      assert is_binary(text)
    end
  end
end
