defmodule CrucibleFramework.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children =
      []
      |> maybe_add_repo()

    opts = [strategy: :one_for_one, name: CrucibleFramework.Supervisor]
    Supervisor.start_link(children, opts)
  end

  defp maybe_add_repo(children) do
    if Application.get_env(:crucible_framework, :enable_repo, true) do
      [CrucibleFramework.Repo | children]
    else
      children
    end
  end
end
