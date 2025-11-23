defmodule Crucible.Thinker.Validation.Schema do
  @moduledoc """
  Validates CLAIM[c*] structure format.

  Parses and validates claims in the format:
  `CLAIM[cN]: <claim text> (citing <doc_id>)`
  """

  @claim_pattern ~r/CLAIM\[c(\d+)\]:\s*(.+?)\s*\(citing\s+(\d+)\)/

  @type claim :: %{
          index: pos_integer(),
          text: String.t(),
          doc_id: pos_integer()
        }

  @doc """
  Parses output text to extract structured claims.

  ## Examples

      iex> output = "CLAIM[c1]: Test claim (citing 123)"
      iex> [claim] = Crucible.Thinker.Validation.Schema.parse(output)
      iex> claim.index
      1
      iex> claim.text
      "Test claim"
      iex> claim.doc_id
      123

  """
  @spec parse(String.t()) :: [claim()]
  def parse(output) when is_binary(output) do
    @claim_pattern
    |> Regex.scan(output, capture: :all_but_first)
    |> Enum.map(fn [index, text, doc_id] ->
      %{
        index: String.to_integer(index),
        text: String.trim(text),
        doc_id: String.to_integer(doc_id)
      }
    end)
  end

  @doc """
  Validates a parsed claim structure.

  Returns true if:
  - index > 0
  - text is not empty
  - doc_id > 0

  ## Examples

      iex> claim = %{index: 1, text: "Valid", doc_id: 123}
      iex> Crucible.Thinker.Validation.Schema.validate(claim)
      true

      iex> claim = %{index: 0, text: "Invalid", doc_id: 123}
      iex> Crucible.Thinker.Validation.Schema.validate(claim)
      false

  """
  @spec validate(claim()) :: boolean()
  def validate(%{index: index, text: text, doc_id: doc_id}) do
    index > 0 and
      String.length(text) > 0 and
      doc_id > 0
  end

  @doc """
  Formats a claim back to standard CLAIM[c*] format.

  ## Examples

      iex> claim = %{index: 1, text: "Test", doc_id: 456}
      iex> Crucible.Thinker.Validation.Schema.format(claim)
      "CLAIM[c1]: Test (citing 456)"

  """
  @spec format(claim()) :: String.t()
  def format(%{index: index, text: text, doc_id: doc_id}) do
    "CLAIM[c#{index}]: #{text} (citing #{doc_id})"
  end
end
