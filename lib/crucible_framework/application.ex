defmodule CrucibleFramework.Application do
  @moduledoc """
  OTP application for CrucibleFramework.

  Note: The Repo is NOT started automatically. Host applications should:
  1. Configure the repo: `config :crucible_framework, repo: MyApp.Repo`
  2. Start their own Repo in their supervision tree

  For backwards compatibility, set `start_repo: true` to auto-start
  `CrucibleFramework.Repo` (requires database config).
  """

  use Application

  @impl true
  def start(_type, _args) do
    children =
      []
      |> maybe_add_legacy_repo()

    opts = [strategy: :one_for_one, name: CrucibleFramework.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Legacy support: only start internal Repo if explicitly enabled
  # New pattern: host app provides repo via config :crucible_framework, repo: MyApp.Repo
  defp maybe_add_legacy_repo(children) do
    if Application.get_env(:crucible_framework, :start_repo, false) do
      [CrucibleFramework.Repo | children]
    else
      children
    end
  end
end
