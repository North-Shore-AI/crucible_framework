defmodule Crucible.IR.GuardrailConfig do
  @derive {Jason.Encoder, only: [:profiles, :options]}
  @moduledoc """
  Guardrail configuration (e.g., LlmGuard profiles and options).
  """

  @type t :: %__MODULE__{
          profiles: [atom()],
          options: map()
        }

  defstruct profiles: [],
            options: %{}
end
