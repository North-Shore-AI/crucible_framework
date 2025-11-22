defmodule Crucible.Tinkex.API.Schema do
  @moduledoc """
  OpenAPI schema for the CNS-facing Crucible Tinkex overlay.

  Kept in code to avoid drifting from the actual controllers while remaining
  framework-agnostic.
  """

  @doc """
  Returns a minimal OpenAPI 3.1 map describing the REST surface.
  """
  @spec spec() :: map()
  def spec do
    %{
      openapi: "3.1.0",
      info: %{
        title: "Crucible Tinkex Overlay",
        version: "0.1.0",
        description: "REST contract for CNS-oriented LoRA runs"
      },
      paths: %{
        "/v1/jobs" => %{
          post: %{
            summary: "Submit training/evaluation job",
            security: [%{"bearerAuth" => []}],
            requestBody: %{
              required: true,
              content: %{
                "application/json" => %{
                  schema: %{
                    type: "object",
                    required: ["dataset_manifest"],
                    properties: %{
                      name: %{type: "string"},
                      dataset_manifest: %{type: "string"},
                      hyperparams: %{type: "object"},
                      metadata: %{type: "object"},
                      type: %{type: "string", enum: ["training", "evaluation"]}
                    }
                  }
                }
              }
            },
            responses: %{
              "200" => %{
                description: "Job accepted",
                content: %{
                  "application/json" => %{
                    schema: %{
                      type: "object",
                      properties: %{
                        job_id: %{type: "string"},
                        status: %{type: "string"},
                        stream_token: %{type: "string"},
                        artifacts_path: %{type: "string"}
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "/v1/jobs/{id}" => %{
          get: %{
            summary: "Fetch job status",
            parameters: [%{name: "id", in: "path", required: true, schema: %{type: "string"}}],
            security: [%{"bearerAuth" => []}],
            responses: %{
              "200" => %{
                description: "Job details",
                content: %{
                  "application/json" => %{
                    schema: %{
                      type: "object",
                      properties: %{
                        job_id: %{type: "string"},
                        status: %{type: "string"},
                        artifacts_path: %{type: "string"},
                        spec: %{type: "object"}
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "/v1/jobs/{id}/stream" => %{
          get: %{
            summary: "Stream telemetry via SSE/WebSocket",
            parameters: [
              %{name: "id", in: "path", required: true, schema: %{type: "string"}},
              %{
                name: "X-Stream-Token",
                in: "header",
                required: true,
                schema: %{type: "string"}
              }
            ],
            security: [%{"bearerAuth" => []}],
            responses: %{"101" => %{description: "Switching Protocols / WebSocket upgrade"}}
          }
        },
        "/v1/jobs/{id}/cancel" => %{
          post: %{
            summary: "Cancel a job",
            parameters: [%{name: "id", in: "path", required: true, schema: %{type: "string"}}],
            security: [%{"bearerAuth" => []}],
            responses: %{"202" => %{description: "Cancellation requested"}}
          }
        }
      },
      components: %{
        securitySchemes: %{
          "bearerAuth" => %{
            type: "http",
            scheme: "bearer",
            bearerFormat: "Token"
          }
        }
      }
    }
  end
end
