# RFC: Supporting vLLM as a Rollout Backend in Slime

- **Author**: \<your name\>
- **Status**: Phase 1 Done
- **Audience**: Slime rollout/runtime maintainers, RL training maintainers
- **Last Updated**: 2026-03-03

## 1. Summary

This RFC proposes adding **vLLM** as a first-class rollout backend in Slime while keeping the existing SGLang behavior unchanged and the GRPO training pipeline unaffected.

Core design principles:

1. Define backend-agnostic rollout request/response contracts (`RolloutBackendRequest`/`RolloutBackendResponse`).
2. Isolate protocol differences through backend adapters (`SGLangClient`, `VLLMClient`).
3. Apply explicit capability gating for non-equivalent features (abort, routed experts, prompt logprobs).
4. Managed mode -- Slime manages the vLLM process lifecycle within Ray, on par with the SGLang path.
5. Weight synchronization leverages vLLM's native weight transfer API, automatically selecting the backend based on deployment mode:
   - **Colocate mode**: CUDA IPC (`IPCWeightTransferEngine`) -- zero-copy via shared GPU memory.
   - **Non-colocate mode**: NCCL broadcast (`NCCLWeightTransferEngine`) -- direct GPU transfer.

**Current status**: Phase 1 is complete and verified. GRPO training runs successfully with Qwen2.5-0.5B + GSM8K on 4 GPUs in colocate mode.

## 2. Motivation

Slime currently assumes SGLang behavior in multiple places:

- Rollout generation response format (`meta_info.finish_reason.type`, `output_token_logprobs`, optional `routed_experts`)
- Router control-plane interactions (`/workers`, `/list_workers`, `/abort_request`)
- Ray-based rollout server startup flow

Supporting vLLM is therefore not just "swapping a URL" -- it requires compatibility layers on both the data plane and control plane.

## 3. Goals and Non-Goals

### Goals

- Add `--rollout-backend {sglang,vllm}`
- Keep SGLang as the default, unchanged
- Keep the training-side interface stable
- Support GRPO rollout with explicit compatibility semantics and observability
- **Support colocate mode** (training and inference share GPUs, weight sync via CUDA IPC)

### Non-Goals (Current Phase)

- Full feature parity with all SGLang capabilities
- R3 routing replay on vLLM
- Multi-instance vLLM + router load balancing

## 4. Architecture and Interface Changes

### 4.1 Architecture Changes

#### End-to-End Architecture Diagram

```text
                          +--------------------+
                          |    TrainerLoop     |
                          +---------+----------+
                                    |
              +---------------------+---------------------+
              | (generate)                    (update_weights)
              v                                           v
    +--------------------+              +------------------------------------+
    |  RolloutFunction   |              | weight updater (auto-selected)     |
    |  (sglang_rollout)  |              |  colocate → UpdateWeightFromTensor |
    +---------+----------+              |  otherwise → ...FromDistributed    |
              |                         +---------------+--------------------+
              v                                         |
  +-------------------------------+                     |
  | RolloutBackendRequest         |                     |
  +---------------+---------------+                     |
                  |                                     v
                  v                       +----------------------------+
     +--------------------------+         |  VLLMEngine (Ray actor)    |
     |  RolloutBackendClient    |         |  weight transfer backend:  |
     +------------+-------------+         |   colocate → IPC           |
                  |                       |   otherwise → NCCL         |
    +-------------+-------------+         +-------------+--------------+
    |                           |                       |
    v                           v                       v
+------------+          +-----------+           +---------------+
| SGLang     |          | VLLM      |           | vLLM server   |
| Client     |          | Client    |           | /update_weights|
+-----+------+          +-----+-----+          | /pause /resume |
      |                       |                 | /sleep /wake_up|
      v                       v                 +---------------+
 +-----------+        +-------------+
 | SGLang    |        | vLLM server |
 | Router    |        | /v1/compl.  |
 +-----------+        +-------------+
       \                     /
        \                   /
         v                 v
  +-----------------------------------+
  | RolloutBackendResponse            |
  | (text, token_ids, token_logprobs, |
  |  finish_reason, backend_raw)      |
  +----------------+------------------+
                   |
                   v
  +-----------------------------------+
  | SampleUpdate + Training Pipeline  |
  | (backend-agnostic consumption)    |
  +-----------------------------------+
```

#### Component Responsibility Comparison

| Component | Before RFC | After RFC |
|---|---|---|
| `RolloutFunction` | Generic rollout + SGLang protocol details | Generic rollout orchestration only |
| Backend protocol layer | Implicit in rollout logic | Explicit `RolloutBackendClient` adapter |
| `SGLangClient` | Did not exist (logic scattered) | Owns all SGLang request/response/control-plane details |
| `VLLMClient` | Did not exist | Owns vLLM `/v1/completions` request/response mapping |
| Training/sample pipeline | Indirectly consumed SGLang-format fields | Consumes unified contract fields, backend-agnostic |
| Weight sync | SGLang IPC or NCCL only | Auto-adapts: SGLang IPC / vLLM IPC / vLLM NCCL |

#### Control-Plane Behavior Split

| Control-plane behavior | SGLang path | vLLM path |
|---|---|---|
| Worker registration/discovery API | Supported | Not needed |
| Worker-level abort | Supported (`/abort_request`) | Degraded to timeout/cancel strategy |
| Routed experts replay | Supported | Explicitly unsupported (capability-gated) |
| Health check | Existing SGLang/SlimeRouter | `GET /health` direct connection |
| Memory management (sleep/wake) | `release/resume_memory_occupation` | `POST /sleep` / `POST /wake_up` |

#### A) Rollout Backend Abstraction Layer

- `RolloutBackendClient` (`slime/rollout/backends/base_client.py`): defines `generate()` + `capabilities` abstract interface
- `BackendCapabilities` dataclass: declares `supports_abort`, `supports_routed_experts`, `supports_prompt_logprobs`
- `SGLangClient` (extracted from existing code), `VLLMClient` (new)

**Impact**: Rollout logic calls a unified interface and no longer directly embeds SGLang HTTP semantics.

#### B) Rollout Execution Path

- `sglang_rollout.generate` constructs `RolloutBackendRequest` and consumes `RolloutBackendResponse`
- Selects `SGLangClient` or `VLLMClient` based on `--rollout-backend`
- `VLLMClient` calls vLLM's OpenAI-compatible `/v1/completions` endpoint

**Impact**: Training-side logic remains stable; backend details are encapsulated in adapters.

#### C) RolloutManager Startup Path

- Existing path (SGLang): unchanged
- vLLM path: `_start_vllm_rollout_servers()` creates a `VLLMEngine` Ray actor that starts and manages a local vLLM server process

**Impact**: vLLM gets the same lifecycle management as SGLang.

#### D) Weight Synchronization

Weight synchronization supports two modes, automatically selected based on deployment:

**Selection logic** (`actor.py`):
```python
update_weight_cls = UpdateWeightFromTensor if args.colocate else UpdateWeightFromDistributed
```
Identical to the SGLang selection logic -- backend-agnostic.

##### D.1) Colocate Mode: CUDA IPC

The vLLM server starts with `--weight-transfer-config '{"backend": "ipc"}'`.

Call chain:
```text
UpdateWeightFromTensor.update_weights()
  → engine.pause_generation.remote()           # VLLMEngine → POST /pause?mode=abort
  → _send_to_colocated_vllm_engine()
      → each training rank:
          reduce_tensor(tensor)                 # create CUDA IPC handle
          {gpu_uuid: ipc_handle}                # keyed by physical GPU UUID
      → dist.gather_object (Gloo)              # collect handles from all TP ranks
      → pickle + base64 encode
      → engine.update_weights_from_tensor.remote()
          → VLLMEngine → POST /update_weights
            { "update_info": {
                "names": [...],
                "dtype_names": [...],
                "shapes": [...],
                "ipc_handles_pickled": "base64..."
            }}
          → vLLM IPCWeightTransferEngine.receive_weights()
            → each TP worker looks up its IPC handle by GPU UUID
            → func(*args) reconstructs tensor → load_weights()
  → engine.continue_generation.remote()        # VLLMEngine → POST /resume
```

Key points:
- The training process and vLLM workers share the same physical GPU; CUDA IPC enables zero-copy weight transfer
- For TP>1, IPC handles from all training ranks are merged via Gloo gather; each parameter contains a mapping of all GPU UUIDs
- The training side must keep tensor references alive until `ray.get()` returns (vLLM has finished reading)

##### D.2) Non-Colocate Mode: NCCL Broadcast

The vLLM server starts with `--weight-transfer-config '{"backend": "nccl"}'`.

Call chain:
```text
UpdateWeightFromDistributed.update_weights()
  → engine.init_weights_update_group.remote()  # VLLMEngine → POST /init_weight_transfer_engine
      → vLLM NCCLWeightTransferEngine initializes StatelessProcessGroup + PyNcclCommunicator
  → training side: NCCLWeightTransferEngine.trainer_init()
      → establishes a matching NCCL communicator with vLLM
  → PyNcclCommunicator.broadcast(tensor, src=0)  # direct GPU transfer to vLLM workers
  → engine.update_weights_from_distributed.remote()
      → VLLMEngine → POST /update_weights
        { "update_info": { "names": [...], "dtype_names": [...], "shapes": [...] }}
      → vLLM NCCLWeightTransferEngine.receive_weights()
```

Key points:
- The training side uses vLLM's `NCCLWeightTransferEngine.trainer_init()` to initialize NCCL, compatible with vLLM's internal `StatelessProcessGroup` + `PyNcclCommunicator`
- Cannot use `torch.distributed.init_process_group` (incompatible with vLLM)

#### VLLMEngine Endpoint Mapping

`VLLMEngine` exposes method signatures compatible with `SGLangEngine`. Internal HTTP endpoint mapping:

| Method | VLLMEngine internal HTTP call |
|---|---|
| `pause_generation()` | `POST /pause?mode=abort` |
| `flush_cache()` | no-op |
| `continue_generation()` | `POST /resume` |
| `init_weights_update_group(...)` | `POST /init_weight_transfer_engine` |
| `update_weights_from_distributed(...)` | `POST /update_weights` (NCCL mode) |
| `update_weights_from_tensor(...)` | `POST /update_weights` (IPC mode) |
| `release_memory_occupation()` | `POST /sleep?level=1&mode=abort` |
| `resume_memory_occupation()` | `POST /wake_up` |
| `health_generate()` | `GET /health` |
| `shutdown()` | `process.terminate()` |

**Impact**: Training-side code (`UpdateWeightFromTensor`, `UpdateWeightFromDistributed`, actor code) uses a unified interface, unaware of the specific backend.

#### E) Router Decision

vLLM does not use a router in the current phase:
- Single vLLM instance; `VLLMClient` connects directly to the local vLLM server port
- SGLang Model Gateway only supports SGLang workers, not applicable
- SlimeRouter is only needed for R3 / radix-tree caching scenarios

### 4.2 Interface Changes

#### A) New CLI Arguments

- `--rollout-backend {sglang,vllm}` -- select rollout backend
- `--vllm-base-url` -- manually specify vLLM server address (not needed when auto-managed)
- `--vllm-model` -- model path for vLLM to load (defaults to `--hf-checkpoint`)
- `--vllm-max-retries` -- max retries for generation requests
- `--vllm-enforce-eager` -- disable CUDA graph (default True)

#### B) New Internal Unified Interface

Added to `slime/rollout/base_types.py`:

- `RolloutBackendRequest`: input_ids, sampling_params, return_logprob, return_routed_experts, image_data, session_id
- `RolloutBackendResponse`: text, output_token_ids, output_token_logprobs, finish_reason (`stop|length|abort`), prompt_tokens, completion_tokens, backend_raw, routed_experts

#### C) Capability Gating

`BackendCapabilities` dataclass declares capabilities per backend:

| Capability | SGLangClient | VLLMClient |
|---|---|---|
| `supports_abort` | True | False |
| `supports_routed_experts` | True | False |
| `supports_prompt_logprobs` | True | False |

Unsupported features are explicitly gated, logged, or fail fast at call time.

## 5. File-Level Change List

### New Files

| File | Description |
|---|---|
| `slime/rollout/backends/base_client.py` | `RolloutBackendClient` abstract interface + `BackendCapabilities` |
| `slime/rollout/backends/sglang_client.py` | SGLang rollout client (extracted from existing code) |
| `slime/rollout/backends/vllm_client.py` | vLLM rollout client (`/v1/completions` adapter) |
| `slime/rollout/backends/__init__.py` | Adapter exports |
| `slime/backends/vllm_utils/vllm_engine.py` | `VLLMEngine` Ray actor (process management + weight sync) |
| `run-qwen2.5-0.5B-vllm.sh` | vLLM validation script (Qwen2.5-0.5B + GSM8K + colocate) |

### Modified Files

| File | Description |
|---|---|
| `slime/rollout/base_types.py` | Added `RolloutBackendRequest` / `RolloutBackendResponse` |
| `slime/rollout/sglang_rollout.py` | Uses backend adapter instead of hard-coded SGLang protocol |
| `slime/utils/arguments.py` | Added `--rollout-backend`, vLLM arguments; sets sglang alias defaults for vLLM |
| `slime/ray/rollout.py` | `_start_vllm_rollout_servers()` to start VLLMEngine |
| `slime/backends/megatron_utils/actor.py` | Weight updater selection: `colocate → UpdateWeightFromTensor` (backend-agnostic) |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | Added `_send_to_colocated_vllm_engine()` (CUDA IPC path) |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | vLLM NCCL compatibility: `NCCLWeightTransferEngine.trainer_init()` + `PyNcclCommunicator.broadcast()` |

### Unchanged Files

| File | Description |
|---|---|
| `train.py` | Training loop is backend-agnostic |
| SGLang-related files | SGLang path is completely unaffected |

### Not Used (Differences from Initial RFC Draft)

- **No `worker_extension.py`**: vLLM's native weight transfer API fully meets requirements; no custom WorkerExtension needed
- **No `collective_rpc`**: Uses vLLM's `/init_weight_transfer_engine` and `/update_weights` endpoints

## 6. Capability Compatibility Matrix

| Capability | SGLang | vLLM | Handling |
|---|---|---|---|
| Token-level response logprobs | Supported | Supported (`choice.logprobs.token_logprobs`) | Adapter normalization |
| Output token IDs | `meta_info` | `choice.token_ids` | Adapter normalization |
| Finish reason | `stop\|length\|abort` | `stop\|length` + others | Canonical mapping |
| Worker-level abort | Supported | Not supported | Timeout/cancel degradation |
| Prompt logprobs (OPD) | Supported | Partial | Capability-gated |
| Routed experts (R3) | Supported | Not supported | Explicit gate |
| Colocate (GPU sharing) | Supported (FlattenedTensorBucket IPC) | Supported (CUDA IPC handles) | Native IPC per backend |
| Memory management | `release/resume_memory_occupation` | `POST /sleep` / `POST /wake_up` | Unified interface |
| `include_stop_str_in_output` | `no_stop_trim` | `include_stop_str_in_output` | Parameter mapping |

## 7. Phased Plan

### Phase 1: Managed vLLM GRPO Training ✅ Done

**Goal**: Qwen2.5-0.5B GRPO training on 4 GPUs in colocate mode with GSM8K dataset, running successfully.

Completed deliverables:
- `VLLMEngine` Ray actor: process lifecycle management, `/pause`/`/resume`/`/sleep`/`/wake_up`
- `VLLMClient` rollout adapter: `/v1/completions` request mapping, logprob/finish_reason/token_ids normalization
- `RolloutBackendRequest`/`Response` unified contract
- `SGLangClient` extraction
- `_start_vllm_rollout_servers` startup path
- Argument parsing branching (`--rollout-backend vllm`, sglang alias defaults)
- **Colocate mode**: CUDA IPC weight transfer (`_send_to_colocated_vllm_engine` → `IPCWeightTransferEngine`)
- **Non-colocate mode**: NCCL weight transfer (`NCCLWeightTransferEngine.trainer_init()` → `PyNcclCommunicator.broadcast()`)
- Validation script: `run-qwen2.5-0.5B-vllm.sh`

Technical decisions:
- No router (single vLLM instance, direct connection)
- Colocate weight sync: CUDA IPC (training and vLLM share GPU memory, zero-copy)
- Non-colocate weight sync: vLLM native NCCL (`StatelessProcessGroup` + `PyNcclCommunicator`)
- Rollout function: reuses `sglang_rollout.py` (backend-agnostic orchestration)
- Training-side weight updater selection is identical to SGLang (`colocate → UpdateWeightFromTensor`)

Acceptance results:
- ✅ Qwen2.5-0.5B + GSM8K + 4 GPU colocate training runs successfully
- ✅ Weight sync correctness: rollout outputs change with each training update
- ✅ Existing SGLang path unaffected

### Phase 2: Multi-Instance vLLM + Async Training

**Goal**: Support multi-instance vLLM rollout with [vllm-project/router](https://github.com/vllm-project/router) for load balancing, and verify async training (`train_async.py`) on larger models.

Key deliverables:
- Integrate [vllm-project/router](https://github.com/vllm-project/router) as the vLLM-side load balancer
- `start_rollout_servers` launches N `VLLMEngine` actors + one vllm-router process
- `VLLMClient` generation requests routed through vllm-router instead of direct connection
- Verify async training (`train_async.py`) correctness with the vLLM backend

Acceptance criteria:
- Multi-instance (e.g., 2-4 vLLM workers) rollout stable over 20+ rollout steps
- Async training (`train_async.py`) with no hangs or weight sync race conditions
- Throughput improvement compared to single-instance baseline
- Larger model verification (e.g., Qwen3-4B, TP=2)

### Phase 3 (Future): Advanced Features

- Abort/cancel strategy refinement
- Prompt logprob support (OPD scenarios)
- Deterministic computation verification
- Performance benchmarks (vLLM vs SGLang throughput comparison)

## 8. Validation Strategy

### Completed (Phase 1)

- ✅ End-to-end GRPO training: `run-qwen2.5-0.5B-vllm.sh` (Qwen2.5-0.5B, GSM8K, 4 GPUs, colocate)
- ✅ Weight sync correctness verification
- ✅ SGLang path non-regression

### Remaining

- Unit tests: finish reason normalization, token/logprob alignment, capability-gated behavior
- Integration tests: SGLang vs vLLM response schema comparison
- Stress tests: higher concurrency / latency pressure

## 9. Key Implementation Challenges and Solutions

### 9.1 vLLM NCCL Incompatibility

**Problem**: vLLM internally uses `StatelessProcessGroup` + `PyNcclCommunicator` for NCCL communication, which is incompatible with `torch.distributed.init_process_group`.

**Solution**: The training side uses `NCCLWeightTransferEngine.trainer_init()` to initialize NCCL, ensuring a matching communicator is established with the vLLM side.

### 9.2 vLLM Logprobs Format Differences

**Problem**: vLLM's `/v1/completions` logprobs format (`choice.token_ids`, `choice.logprobs.token_logprobs`) differs from SGLang's.

**Solution**: `VLLMClient` performs explicit mapping, converting vLLM's format into the unified `RolloutBackendResponse` format.
