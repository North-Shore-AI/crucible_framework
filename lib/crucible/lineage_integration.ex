defmodule Crucible.LineageIntegration do
  @moduledoc """
  Emits LineageIR-compatible spans and artifacts for pipeline stage execution.

  Emission defaults to storing spans and artifacts in `context.assigns` so
  callers can inspect lineage data without requiring a sink integration.
  """

  alias Crucible.Context
  alias CrucibleIR.StageDef

  @source "crucible_framework"

  @type span :: map()
  @type artifact :: map()

  @spec init_trace(Context.t(), keyword()) :: Context.t()
  def init_trace(%Context{} = ctx, opts \\ []) do
    enabled? = Keyword.get(opts, :enable_lineage, true)

    if enabled? do
      trace_id = Keyword.get(opts, :trace_id, Map.get(ctx.telemetry_context, :trace_id))
      trace_id = trace_id || Ecto.UUID.generate()

      telemetry_context =
        ctx.telemetry_context
        |> Map.put(:trace_id, trace_id)
        |> Map.put(:run_id, ctx.run_id)
        |> Map.put(:experiment_id, ctx.experiment_id)
        |> Map.put(:lineage_source, @source)
        |> Map.put(:lineage_enabled, true)

      ctx
      |> put_telemetry_context(telemetry_context)
      |> ensure_assign(:lineage_spans, [])
      |> ensure_assign(:lineage_artifacts, [])
      |> ensure_assign(:lineage_active_spans, %{})
    else
      put_telemetry_context(ctx, Map.put(ctx.telemetry_context, :lineage_enabled, false))
    end
  end

  @spec enabled?(Context.t()) :: boolean()
  def enabled?(%Context{} = ctx) do
    Map.get(ctx.telemetry_context, :lineage_enabled, true)
  end

  @spec restore_tracking(Context.t(), Context.t()) :: Context.t()
  def restore_tracking(%Context{} = ctx, %Context{} = previous_ctx) do
    telemetry_context =
      previous_ctx.telemetry_context
      |> Map.merge(ctx.telemetry_context)

    ctx
    |> put_telemetry_context(telemetry_context)
    |> ensure_assign(:lineage_spans, previous_ctx.assigns[:lineage_spans] || [])
    |> ensure_assign(:lineage_artifacts, previous_ctx.assigns[:lineage_artifacts] || [])
    |> ensure_assign(:lineage_active_spans, previous_ctx.assigns[:lineage_active_spans] || %{})
  end

  @spec emit_stage_start(Context.t(), StageDef.t()) :: Context.t()
  def emit_stage_start(%Context{} = ctx, %StageDef{} = stage_def) do
    if enabled?(ctx) do
      trace_id = Map.get(ctx.telemetry_context, :trace_id) || Ecto.UUID.generate()
      ctx = put_telemetry_context(ctx, Map.put(ctx.telemetry_context, :trace_id, trace_id))
      span_id = Ecto.UUID.generate()
      step_id = step_id_for(stage_def) || Ecto.UUID.generate()
      started_at = DateTime.utc_now()

      span = %{
        id: span_id,
        trace_id: trace_id,
        run_id: ctx.run_id,
        step_id: step_id,
        name: Atom.to_string(stage_def.name),
        kind: "stage",
        status: "running",
        attributes: build_attributes(ctx, stage_def),
        metrics: %{},
        started_at: started_at,
        finished_at: nil
      }

      active =
        ctx.assigns
        |> Map.get(:lineage_active_spans, %{})
        |> Map.put(stage_def.name, span)

      ctx
      |> Context.assign(:lineage_active_spans, active)
      |> ensure_assign(:lineage_spans, Map.get(ctx.assigns, :lineage_spans, []))
    else
      ctx
    end
  end

  @spec emit_stage_complete(Context.t(), StageDef.t(), map(), list()) :: Context.t()
  def emit_stage_complete(%Context{} = ctx, %StageDef{} = stage_def, metrics, artifacts) do
    if enabled?(ctx) do
      {span, ctx} = pop_active_span(ctx, stage_def.name)

      span =
        span
        |> Map.put(:status, "succeeded")
        |> Map.put(:metrics, metrics || %{})
        |> Map.put(:finished_at, DateTime.utc_now())

      ctx
      |> append_span(span)
      |> emit_artifacts(span, stage_def, artifacts)
    else
      ctx
    end
  end

  @spec emit_stage_failed(Context.t(), StageDef.t(), term()) :: Context.t()
  def emit_stage_failed(%Context{} = ctx, %StageDef{} = stage_def, reason) do
    if enabled?(ctx) do
      {span, ctx} = pop_active_span(ctx, stage_def.name)
      {error_type, error_message} = format_error(reason)

      span =
        span
        |> Map.put(:status, "failed")
        |> Map.put(:error_type, error_type)
        |> Map.put(:error_message, error_message)
        |> Map.put(:finished_at, DateTime.utc_now())

      append_span(ctx, span)
    else
      ctx
    end
  end

  defp append_span(%Context{} = ctx, span) do
    spans = Map.get(ctx.assigns, :lineage_spans, []) || []
    Context.assign(ctx, :lineage_spans, spans ++ [span])
  end

  defp emit_artifacts(%Context{} = ctx, _span, _stage_def, []), do: ctx

  defp emit_artifacts(%Context{} = ctx, span, %StageDef{} = stage_def, artifacts) do
    artifact_entries =
      artifacts
      |> Enum.map(&build_artifact(&1, span, ctx, stage_def))

    existing = Map.get(ctx.assigns, :lineage_artifacts, []) || []
    Context.assign(ctx, :lineage_artifacts, existing ++ artifact_entries)
  end

  defp build_artifact({key, value}, span, %Context{} = ctx, %StageDef{} = stage_def) do
    {uri, metadata} = normalize_artifact_value(value)

    %{
      id: Ecto.UUID.generate(),
      trace_id: span.trace_id,
      span_id: span.id,
      run_id: ctx.run_id,
      step_id: span.step_id,
      type: Atom.to_string(key),
      uri: uri,
      checksum: extract_field(value, :checksum),
      size_bytes: extract_field(value, :size_bytes) || extract_field(value, :size),
      mime_type: extract_field(value, :mime_type),
      metadata: Map.merge(metadata, %{stage: stage_def.name}),
      created_at: DateTime.utc_now()
    }
  end

  defp normalize_artifact_value(value) when is_binary(value), do: {value, %{}}

  defp normalize_artifact_value(value) when is_map(value) do
    uri =
      Map.get(value, :uri) ||
        Map.get(value, :path) ||
        Map.get(value, :location) ||
        Map.get(value, :url)

    metadata =
      value
      |> Map.drop([:uri, :path, :location, :url, :checksum, :size, :size_bytes, :mime_type])
      |> sanitize_metadata()

    {uri, metadata}
  end

  defp normalize_artifact_value(value), do: {nil, %{value: inspect(value)}}

  defp sanitize_metadata(map) when is_map(map) do
    map
    |> Enum.map(fn {key, val} -> {key, sanitize_value(val)} end)
    |> Enum.into(%{})
  end

  defp sanitize_value(value)
       when is_pid(value) or is_reference(value) or is_function(value) or is_port(value),
       do: inspect(value)

  defp sanitize_value(value), do: value

  defp extract_field(value, key) when is_map(value), do: Map.get(value, key)
  defp extract_field(_value, _key), do: nil

  defp pop_active_span(%Context{} = ctx, stage_name) do
    active = Map.get(ctx.assigns, :lineage_active_spans, %{})
    span = Map.get(active, stage_name) || build_fallback_span(ctx, stage_name)
    updated_active = Map.delete(active, stage_name)

    {span, Context.assign(ctx, :lineage_active_spans, updated_active)}
  end

  defp build_fallback_span(%Context{} = ctx, stage_name) do
    %{
      id: Ecto.UUID.generate(),
      trace_id: Map.get(ctx.telemetry_context, :trace_id),
      run_id: ctx.run_id,
      step_id: Ecto.UUID.generate(),
      name: Atom.to_string(stage_name),
      kind: "stage",
      status: "unknown",
      attributes: %{},
      metrics: %{},
      started_at: DateTime.utc_now(),
      finished_at: nil
    }
  end

  defp step_id_for(%StageDef{options: options}) when is_map(options) do
    options[:step_id] || options[:plan_step_id] || options[:step_key]
  end

  defp step_id_for(_), do: nil

  defp build_attributes(%Context{} = ctx, %StageDef{} = stage_def) do
    stage_opts = stage_def.options || %{}

    %{
      stage: stage_def.name,
      options: stage_opts,
      experiment_id: ctx.experiment_id,
      plan_id: stage_opts[:plan_id],
      depends_on: stage_opts[:depends_on]
    }
    |> Enum.reject(fn {_key, value} -> is_nil(value) end)
    |> Enum.into(%{})
  end

  defp format_error(%{__struct__: _} = error) do
    {error.__struct__ |> Module.split() |> List.last(), Exception.message(error)}
  end

  defp format_error(error) when is_atom(error), do: {"error", Atom.to_string(error)}
  defp format_error(error) when is_binary(error), do: {"error", error}
  defp format_error(error), do: {"error", inspect(error)}

  defp ensure_assign(%Context{} = ctx, key, default) do
    if Map.has_key?(ctx.assigns, key) do
      ctx
    else
      Context.assign(ctx, key, default)
    end
  end

  defp put_telemetry_context(%Context{} = ctx, telemetry_context) do
    %Context{ctx | telemetry_context: telemetry_context}
  end
end
