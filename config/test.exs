import Config

config :crucible_framework, CrucibleFramework.Repo,
  username: "crucible_dev",
  password: "crucible_dev_pw",
  hostname: "localhost",
  port: 5432,
  database: "crucible_framework_test",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: 10

enable_repo? = System.get_env("CRUCIBLE_DB_ENABLED") == "true"

config :crucible_framework, :enable_repo, enable_repo?

config :logger, level: :warning
