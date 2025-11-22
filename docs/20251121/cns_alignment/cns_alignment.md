# CNS Overlay Integration Blueprint

## Purpose

- Establish Crucible Framework as the single backend that interfaces with Tinkex (job submission, telemetry, checkpoints).
- Provide a clean API/contract so CNS-oriented surfaces (CNS UI, research scripts) can orchestrate LoRA runs without embedding SDK logic.
- Document responsibilities, data flows, and immediate tasks required to expose those services.

## Guiding Principles

1. **Single owner of Tinkex credentials** – only the `crucible_tinkex` OTP app communicates with the SDK.
2. **Contract-first** – expose REST/gRPC endpoints plus Phoenix PubSub topics for telemetry; no UI reaches into internal supervisors.
3. **Reusable artifacts** – adapters, manifests, checkpoints, and telemetry payloads live under `artifacts/crucible/{run_id}` with manifests describing hashes + provenance.
4. **CNS remains a client** – SNS/SNO-specific logic stays out of the framework; it consumes API responses the same way any other research program would.

## Target Architecture

```
CNS UI / other clients
        │
        ▼
Crucible API Gateway (REST/gRPC)
        │
        ▼
Crucible Framework services
  ├─ Experiment Registry
  ├─ Scheduler / Queue
  ├─ Telemetry Broker
  └─ Tinkex Adapter (single point of contact)
        │
        ▼
     Tinkex SDK
```

### API Surfaces

| Endpoint | Function | Notes |
| --- | --- | --- |
| `POST /v1/jobs` | Submit training/eval job with dataset manifest + hyperparams | Returns job_id + stream token |
| `GET /v1/jobs/:id` | Fetch status, metrics, latest checkpoints | Includes pointers to artifacts |
| `GET /v1/jobs/:id/stream` | Server-sent events or WebSocket for telemetry | Emits standardized `[:crucible, :tinkex, ...]` events |
| `POST /v1/jobs/:id/cancel` | Request graceful shutdown | Surfaces result in telemetry |

### Artifact Layout

- `artifacts/crucible/<job_id>/manifest.json` – config digest, dataset hash, timestamps.
- `.../telemetry.jsonl` – append-only log mirroring SSE stream.
- `.../checkpoints/*.bin` – raw Tinkex outputs; metadata includes target model and LoRA rank.

## Immediate Tasks

1. **API layer**
   - [ ] Define OpenAPI schema describing requests/responses and error codes.
   - [ ] Implement Phoenix controller (REST) + WebSocket channel for telemetry streaming.
   - [ ] Add service authentication (API tokens scoped per UI/app).

2. **Scheduler + queue**
   - [ ] Wrap current Tinkex job submission with ETS/DB-backed queue to handle retries and concurrency limits.
   - [ ] Persist job spec + status in `Crucible.Experiments` tables for UI consumption.

3. **Telemetry plumbing**
   - [ ] Ensure `Crucible.Tinkex.Telemetry` broadcasts via both `:telemetry` and Phoenix PubSub.
   - [ ] Normalize payload schema (job_id, step, epoch, loss, quality metrics).

4. **Documentation & samples**
   - [ ] Provide example curl/API client scripts.
   - [ ] Ship reference integration for CNS UI demonstrating job submission + stream consumption.

## Acceptance Criteria

- CNS UI can start/cancel a LoRA training run using only the public API.
- All telemetry visible in CNS UI is sourced from the shared PubSub/WebSocket channel.
- No package other than `crucible_framework` declares a dependency on `tinkex`.

## Implementation Notes (2025-11-21)

- REST/WebSocket scaffolding is implemented in `Crucible.Tinkex.API.Router` (framework-agnostic controller functions), `Crucible.Tinkex.API.Schema` (OpenAPI map), and `Crucible.Tinkex.API.Stream` (SSE/WebSocket helper). Phoenix/Plug wiring is left to the host gateway, per contract-first guidance.
- Telemetry is brokered through `Crucible.Tinkex.TelemetryBroker`, which mirrors `:telemetry` events (job_id, step/epoch, metrics) to stream subscribers using in-process PubSub and signed stream tokens.
- Job submission, queueing, and status persistence live in `Crucible.Tinkex.JobQueue` and `Crucible.Tinkex.JobStore`; jobs carry dataset/hyperparam manifests only. Tinkex credentials remain internal and are never accepted from API callers, satisfying the single-owner requirement.
