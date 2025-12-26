ExUnit.start(exclude: [:integration])

{:ok, _} = Application.ensure_all_started(:crucible_framework)

if Application.get_env(:crucible_framework, :enable_repo, false) do
  Ecto.Adapters.SQL.Sandbox.mode(CrucibleFramework.Repo, :manual)
end

# Mock for Stage behaviour (used in pipeline runner tests)
Mox.defmock(Crucible.StageMock, for: Crucible.Stage)

# Mock for Guardrails adapter
Mox.defmock(Crucible.GuardrailMock, for: Crucible.Stage.Guardrails.Adapter)
