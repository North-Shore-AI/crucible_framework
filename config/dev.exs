import Config

config :crucible_framework, CrucibleFramework.Repo,
  username: "crucible_dev",
  password: "crucible_dev_pw",
  hostname: "localhost",
  port: 5432,
  database: "crucible_framework_dev",
  pool_size: 10

config :tinkex,
  api_key: System.get_env("TINKER_API_KEY"),
  base_url:
    System.get_env(
      "TINKER_BASE_URL",
      "https://tinker.thinkingmachines.dev/services/tinker-prod"
    )
