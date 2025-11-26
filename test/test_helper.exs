ExUnit.start(exclude: [:integration])

{:ok, _} = Application.ensure_all_started(:crucible_framework)

if Application.get_env(:crucible_framework, :enable_repo, false) do
  Ecto.Adapters.SQL.Sandbox.mode(CrucibleFramework.Repo, :manual)
end

Mox.defmock(Crucible.BackendMock, for: Crucible.Backend)
Mox.defmock(Crucible.GuardrailMock, for: Crucible.Stage.Guardrails.Adapter)
Mox.defmock(Crucible.AnalysisMock, for: Crucible.Analysis.Adapter)
Mox.defmock(Crucible.Backend.Tinkex.ClientMock, for: Crucible.Backend.Tinkex.Client)
