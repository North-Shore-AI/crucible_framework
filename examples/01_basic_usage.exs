#!/usr/bin/env elixir

# Basic Usage Example
# This example demonstrates the fundamental features of CrucibleFramework

IO.puts("\n=== Crucible Framework: Basic Usage Example ===\n")

# 1. Get framework information
IO.puts("1. Framework Information")
IO.puts("   Version: #{CrucibleFramework.version()}")
info = CrucibleFramework.info()
IO.puts("   Elixir Version: #{info.elixir_version}")
IO.puts("   OTP Release: #{info.otp_release}")

# 2. List available components
IO.puts("\n2. Available Components")
components = CrucibleFramework.components()
IO.puts("   Total Components: #{length(components)}")

Enum.each(components, fn component ->
  status = CrucibleFramework.component_status(component)
  status_icon = if status == :available, do: "✓", else: "○"
  IO.puts("   #{status_icon} #{component}: #{status}")
end)

# 3. Check loaded components
IO.puts("\n3. Loaded Components")
loaded = CrucibleFramework.loaded_components()

if Enum.empty?(loaded) do
  IO.puts("   No components currently loaded")
  IO.puts("   Note: Components are separate packages that can be installed via Mix")
else
  Enum.each(loaded, fn component ->
    IO.puts("   ✓ #{component}")
  end)
end

IO.puts("\n=== Example Complete ===\n")
