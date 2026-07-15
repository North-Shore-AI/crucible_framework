defmodule Crucible.Pipeline.RunnerLineageTest do
  use ExUnit.Case, async: true

  alias Crucible.Context
  alias Crucible.Pipeline.Runner
  alias CrucibleIR.{BackendRef, Experiment, StageDef}

  defmodule ArtifactStage do
    @behaviour Crucible.Stage

    @impl true
    def run(%Context{} = ctx, _opts) do
      ctx =
        Context.put_artifact(ctx, :report, %{
          uri: "file://reports/report.md",
          format: "markdown"
        })

      {:ok, ctx}
    end

    @impl true
    def describe(_opts) do
      %{
        name: :artifact_stage,
        description: "Adds a report artifact to the context",
        required: [],
        optional: [],
        types: %{}
      }
    end
  end

  defmodule FailingStage do
    @behaviour Crucible.Stage

    @impl true
    def run(%Context{} = _ctx, _opts), do: {:error, :boom}

    @impl true
    def describe(_opts) do
      %{
        name: :failing_stage,
        description: "Fails intentionally",
        required: [],
        optional: [],
        types: %{}
      }
    end
  end

  test "emits lineage spans and artifacts for stage execution" do
    experiment = %Experiment{
      id: "lineage-exp",
      backend: %BackendRef{id: :noop},
      pipeline: [
        %StageDef{name: :artifact_stage, module: ArtifactStage}
      ]
    }

    assert {:ok, ctx} = Runner.run(experiment, persist: false, enable_lineage: true)

    spans = ctx.assigns[:lineage_spans]
    artifacts = ctx.assigns[:lineage_artifacts]

    assert is_list(spans)
    assert is_list(artifacts)
    assert length(spans) == 1
    assert length(artifacts) == 1

    span = List.first(spans)
    artifact = List.first(artifacts)

    assert span.name == "artifact_stage"
    assert span.status == "succeeded"
    assert span.trace_id == ctx.telemetry_context[:trace_id]

    assert artifact.type == "report"
    assert artifact.span_id == span.id
    assert artifact.trace_id == span.trace_id
    assert artifact.step_id == span.step_id
  end

  test "records failed stage spans with error details" do
    experiment = %Experiment{
      id: "lineage-fail-exp",
      backend: %BackendRef{id: :noop},
      pipeline: [
        %StageDef{name: :failing_stage, module: FailingStage}
      ]
    }

    assert {:error, {:failing_stage, :boom, ctx}} =
             Runner.run(experiment, persist: false, enable_lineage: true)

    [span] = ctx.assigns[:lineage_spans]

    assert span.name == "failing_stage"
    assert span.status == "failed"
    assert span.error_type == "error"
    assert span.error_message == "boom"
  end
end
