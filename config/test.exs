import Config

config :crucible_framework, CrucibleFramework.Repo,
  username: "crucible_dev",
  password: "crucible_dev_pw",
  hostname: "localhost",
  port: 5432,
  database: "crucible_framework_test",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: 10

config :crucible_framework, :enable_repo, true

config :tinkex,
  api_key: System.get_env("TINKER_API_KEY", "test-key"),
  base_url: System.get_env("TINKER_BASE_URL", "https://example.invalid")

config :logger, level: :warning
