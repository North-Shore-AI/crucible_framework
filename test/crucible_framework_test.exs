defmodule CrucibleFrameworkTest do
  use ExUnit.Case
  doctest CrucibleFramework

  describe "version/0" do
    test "returns the framework version" do
      version = CrucibleFramework.version()
      assert is_binary(version)
      assert String.match?(version, ~r/^\d+\.\d+\.\d+/)
    end
  end

  describe "components/0" do
    test "returns a list of component atoms" do
      components = CrucibleFramework.components()
      assert is_list(components)
      assert length(components) > 0
      assert Enum.all?(components, &is_atom/1)
    end

    test "includes expected core components" do
      components = CrucibleFramework.components()
      assert :ensemble in components
      assert :hedging in components
      assert :bench in components
      assert :telemetry_research in components
      assert :dataset_manager in components
      assert :causal_trace in components
      assert :research_harness in components
      assert :reporter in components
    end
  end

  describe "component_status/1" do
    test "returns :not_loaded for valid but unavailable components" do
      assert CrucibleFramework.component_status(:ensemble) == :not_loaded
      assert CrucibleFramework.component_status(:hedging) == :not_loaded
    end

    test "returns :unknown for invalid components" do
      assert CrucibleFramework.component_status(:invalid_component) == :unknown
      assert CrucibleFramework.component_status(:not_a_real_lib) == :unknown
    end
  end

  describe "info/0" do
    test "returns a map with framework information" do
      info = CrucibleFramework.info()
      assert is_map(info)
      assert Map.has_key?(info, :version)
      assert Map.has_key?(info, :components)
      assert Map.has_key?(info, :loaded_components)
      assert Map.has_key?(info, :elixir_version)
      assert Map.has_key?(info, :otp_release)
    end

    test "version in info matches version/0" do
      info = CrucibleFramework.info()
      assert info.version == CrucibleFramework.version()
    end

    test "components in info matches components/0" do
      info = CrucibleFramework.info()
      assert info.components == CrucibleFramework.components()
    end

    test "includes system information" do
      info = CrucibleFramework.info()
      assert is_binary(info.elixir_version)
      assert is_binary(info.otp_release)
    end
  end

  describe "loaded_components/0" do
    test "returns a list" do
      loaded = CrucibleFramework.loaded_components()
      assert is_list(loaded)
    end

    test "only includes components that are actually loaded" do
      loaded = CrucibleFramework.loaded_components()
      # Since no actual component libraries are installed, should be empty
      assert loaded == []
    end
  end
end
