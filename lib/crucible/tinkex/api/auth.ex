defmodule Crucible.Tinkex.API.Auth do
  @moduledoc """
  Lightweight API token validation for the Crucible Tinkex overlay.

  The CNS blueprint requires service authentication at the API layer without
  exposing Tinkex credentials. This module enforces that rule by validating
  bearer-style tokens supplied by external clients while keeping the actual
  Tinkex API key contained inside the `crucible_tinkex` application.
  """

  @type headers :: [{String.t(), String.t()}]

  @doc """
  Validates that the caller provided a known API token.

  The accepted tokens are configured under `:crucible_tinkex, :api_tokens`
  as a list of strings. Returns the token that was validated to support
  downstream auditing.
  """
  @spec verify(headers()) :: {:ok, String.t()} | {:error, :unauthorized}
  def verify(headers) when is_list(headers) do
    with {:ok, provided} <- extract_token(headers),
         true <- provided in configured_tokens() do
      {:ok, provided}
    else
      _ -> {:error, :unauthorized}
    end
  end

  defp extract_token(headers) do
    case Enum.find(headers, fn {k, _} -> String.downcase(k) == "authorization" end) do
      {_, "Bearer " <> token} when byte_size(token) > 0 -> {:ok, token}
      _ -> {:error, :missing}
    end
  end

  defp configured_tokens do
    Application.get_env(:crucible_tinkex, :api_tokens, [])
  end
end
