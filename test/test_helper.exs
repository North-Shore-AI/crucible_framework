ExUnit.start(capture_log: true, exclude: [:slow])

Logger.configure(level: :warning)

# Start the telemetry registry for tests
{:ok, _} = Registry.start_link(keys: :unique, name: Crucible.Telemetry.Registry)
