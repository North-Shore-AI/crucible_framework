defmodule Crucible.Tinkex.API.Stream do
  @moduledoc """
  Utilities for exposing telemetry streams over SSE/WebSocket channels.
  """

  @doc """
  Converts mailbox messages into a lazy stream suitable for SSE handlers.
  """
  @spec to_enum((-> :ok), keyword()) :: Enumerable.t()
  def to_enum(subscribe_fun, opts \\ []) when is_function(subscribe_fun, 0) do
    subscribe_fun.()
    timeout = Keyword.get(opts, :timeout, :infinity)

    Stream.resource(
      fn -> :ok end,
      fn
        :ok ->
          receive do
            {:crucible_tinkex_event, job_id, event} -> {[{job_id, event}], :ok}
          after
            timeout -> {:halt, :ok}
          end
      end,
      fn _ -> :ok end
    )
  end
end
