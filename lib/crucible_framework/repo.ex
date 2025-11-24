defmodule CrucibleFramework.Repo do
  use Ecto.Repo,
    otp_app: :crucible_framework,
    adapter: Ecto.Adapters.Postgres
end
