defmodule Mix.Tasks.Crucible.Stages do
  @shortdoc "Lists available Crucible stages and their options"

  @moduledoc """
  Lists available Crucible stages and their options.

  ## Usage

      mix crucible.stages              # List all registered stages
      mix crucible.stages --name bench # Show details for :bench stage
      mix crucible.stages -n validate  # Short form

  ## Examples

      $ mix crucible.stages

      Available Stages:
      =================

      :bench (Crucible.Stage.Bench)
        Description: Statistical benchmarking and hypothesis testing
        Required: (none)
        Optional: tests, alpha, data_source, options

      :validate (Crucible.Stage.Validate)
        Description: Pre-flight validation of pipeline stages
        Required: (none)
        Optional: strict

      $ mix crucible.stages --name bench

      Stage: :bench
      Module: Crucible.Stage.Bench

      Description:
        Statistical benchmarking and hypothesis testing using crucible_bench

      Required Options: (none)

      Optional Options:
        tests        {:list, {:enum, [...]}}  Statistical tests to run
        alpha        :float                    Significance level
        data_source  {:enum, [...]}            Data extraction mode
        options      :map                      Additional test options
  """

  use Mix.Task

  alias Crucible.Registry

  @switches [name: :string]
  @aliases [n: :name]

  @impl true
  def run(args) do
    Mix.Task.run("app.start", [])

    {opts, _, _} = OptionParser.parse(args, switches: @switches, aliases: @aliases)

    case opts[:name] do
      nil -> list_all_stages()
      name -> show_stage_details(String.to_atom(name))
    end
  end

  defp list_all_stages do
    stages = Registry.list_stages_with_schemas()

    if stages == [] do
      Mix.shell().info("No stages registered. Configure :stage_registry in :crucible_framework.")
    else
      Mix.shell().info("\nAvailable Stages:")
      Mix.shell().info("=================\n")
      Enum.each(stages, &print_stage_summary/1)
    end
  end

  defp print_stage_summary({name, mod, schema}) do
    Mix.shell().info(":#{name} (#{inspect(mod)})")
    Mix.shell().info("  Description: #{schema[:description] || "(none)"}")
    print_field_list("Required", schema[:required] || [])
    print_field_list("Optional", schema[:optional] || [])
    Mix.shell().info("")
  end

  defp print_field_list(label, []), do: Mix.shell().info("  #{label}: (none)")

  defp print_field_list(label, fields),
    do: Mix.shell().info("  #{label}: #{Enum.join(fields, ", ")}")

  defp show_stage_details(name) do
    case Registry.stage_schema(name) do
      {:ok, schema} ->
        {:ok, mod} = Registry.stage_module(name)
        print_detailed_schema(name, mod, schema)

      {:error, {:unknown_stage, ^name}} ->
        Mix.shell().info("Unknown stage: :#{name}")
        Mix.shell().info("Run `mix crucible.stages` to see available stages.")

      {:error, {:no_describe_callback, mod}} ->
        Mix.shell().info("Stage #{inspect(mod)} does not implement describe/1")

      {:error, reason} ->
        Mix.shell().info("Error: #{inspect(reason)}")
    end
  end

  defp print_detailed_schema(name, mod, schema) do
    Mix.shell().info("\nStage: :#{name}")
    Mix.shell().info("Module: #{inspect(mod)}")
    print_optional_field("Version", schema[:version])
    print_optional_field("Schema Version", schema[:__schema_version__])

    Mix.shell().info("\nDescription:")
    Mix.shell().info("  #{schema[:description] || "(none)"}")

    types = schema[:types] || %{}
    defaults = schema[:defaults] || %{}

    Mix.shell().info("\nRequired Options:")
    print_options_list(schema[:required] || [], types, %{})

    Mix.shell().info("\nOptional Options:")
    print_options_list(schema[:optional] || [], types, defaults)

    print_extensions(schema[:__extensions__])
    Mix.shell().info("")
  end

  defp print_optional_field(_label, nil), do: :ok
  defp print_optional_field(label, value), do: Mix.shell().info("#{label}: #{value}")

  defp print_options_list([], _types, _defaults), do: Mix.shell().info("  (none)")

  defp print_options_list(keys, types, defaults) do
    Enum.each(keys, fn key ->
      type_str = format_type(types[key])
      default_str = format_default(defaults[key])
      Mix.shell().info("  #{pad(key, 14)} #{type_str}#{default_str}")
    end)
  end

  defp format_default(nil), do: ""
  defp format_default(val), do: " (default: #{inspect(val)})"

  defp print_extensions(nil), do: :ok

  defp print_extensions(extensions) do
    Mix.shell().info("\nExtensions:")

    Enum.each(extensions, fn {key, value} ->
      Mix.shell().info("  #{key}: #{inspect(value, pretty: true, limit: 50)}")
    end)
  end

  defp format_type(nil), do: "(unspecified)"
  defp format_type(type), do: inspect(type)

  defp pad(atom, width) when is_atom(atom) do
    str = Atom.to_string(atom)
    String.pad_trailing(str, width)
  end
end
