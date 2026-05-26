# RFC: Add vLLM as a Rollout Backend in Slime

- **Author**: \<your name\>
- **Status**: Draft
- **Audience**: Slime rollout/runtime maintainers, RL training maintainers
- **Last Updated**: 2026-03-02

## 1. Summary

This RFC proposes adding **vLLM** as a first-class rollout backend in Slime while preserving current SGLang behavior and avoiding regressions in GRPO workflows.

The design is based on:

1. A backend-agnostic rollout request/response contract.
2. Backend adapters (`SGLangClient`, `VLLMClient`) that isolate protocol differences.
3. Capability-aware behavior for non-parity features (abort, routed experts, prompt logprobs).
4. Managed vLLM mode from day one (Slime manages vLLM process lifecycle inside Ray, same as SGLang path).
5. Weight sync via NCCL broadcast (GPU direct transfer, no disk I/O).

## 2. Why this is needed

Slime currently assumes SGLang behavior in multiple places:

- rollout generation response schema (`meta_info.finish_reason.type`, `output_token_logprobs`, optional `routed_experts`)
- router control-plane interactions (`/workers`, `/list_workers`, `/abort_request`)
- rollout server startup flow in Ray

Supporting vLLM is therefore not a URL replacement. It requires a compatibility layer across both data plane and control plane semantics.

## 3. Goals and Non-goals

### Goals

- Add `--rollout-backend {sglang,vllm}`.
- Keep SGLang path unchanged by default.
- Keep trainer/algorithm interfaces stable.
- Support GRPO rollout with explicit compatibility semantics and observability.

### Non-goals (initial phase)

- Full parity for all SGLang-specific features.
- Colocate mode (training and rollout sharing same GPUs).
- R3 routed expert replay on vLLM.
- Multi-instance vLLM with router load balancing.

## 4. Architecture and Interface Changes (Explicit)

This section is the key communication point for Slime maintainers.

### 4.1 Architecture changes

#### End-to-end architecture (string diagram)

```text
                          +--------------------+
                          |     TrainerLoop    |
                          +---------+----------+
                                    |
                                    v
                          +--------------------+
                          |   RolloutFunction  |
                          +---------+----------+
                                    |
                                    v
                    +-------------------------------+
                    | CanonicalRolloutRequest       |
                    | (input_ids, sampling_params,  |
                    |  return_logprob, prompt_text) |
                    +---------------+---------------+
                                    |
                                    v
                       +--------------------------+
                       |   RolloutBackendClient   |
                       +------------+-------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
      +------------------------+          +------------------------+
      |      SGLangClient      |          |       VLLMClient       |
      +-----------+------------+          +------------+-----------+
                  |                                    |
                  v                                    v
   +-------------------------------+      +------------------------------+
   | SGLangRouter or SlimeRouter   |      | SlimeRouter(generic) or      |
   | (SGLang control-plane aware)  |      | direct vLLM endpoint          |
   +---------------+---------------+      +--------------+---------------+
                   |                                     |
                   v                                     v
        +----------------------+             +----------------------+
        |    SGLang workers    |             |    vLLM workers      |
        +----------+-----------+             +----------+-----------+
                   \                                     /
                    \                                   /
                     v                                 v
                    +-----------------------------------+
                    | CanonicalRolloutResponse          |
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

#### Component responsibility map

| Component | Responsibility before RFC | Responsibility after RFC |
|---|---|---|
| `RolloutFunction` | Contains generic rollout + SGLang protocol details | Contains generic rollout orchestration only |
| Backend protocol layer | Implicit in rollout logic | Explicit via `RolloutBackendClient` adapters |
| `SGLangClient` | N/A (scattered logic) | Owns SGLang request/response/control-plane specifics |
| `VLLMClient` | N/A | Owns vLLM request/response mapping and retries |
| Trainer/sample pipeline | Consumes SGLang-shaped fields indirectly | Consumes canonical fields, backend-agnostic |
| Router integration | Mixed generic + SGLang-specific assumptions | SGLang-specific control-plane isolated to SGLang adapter |

#### Control-plane behavior split

| Control-plane behavior | SGLang path | vLLM initial path |
|---|---|---|
| Worker registration/discovery APIs | Supported | Not required in generic path |
| Worker-level abort | Supported (`/abort_request`) | Fallback to timeout/cancel semantics |
| Routed experts replay metadata | Supported | Explicitly unsupported (capability-gated) |
| Health/load-balance routing | Existing SGLang/SlimeRouter behavior | SlimeRouter generic mode or direct endpoint |

#### A) Rollout backend abstraction layer (new)

- Introduce a backend client interface:
  - `RolloutBackendClient`
  - backend capability descriptor
- Add concrete adapters:
  - `SGLangClient` (existing behavior extraction)
  - `VLLMClient` (new)

**Impact**: rollout logic calls a unified backend interface instead of directly embedding SGLang HTTP semantics.

#### B) Rollout execution path refactor

- `sglang_rollout.generate` no longer directly depends on SGLang HTTP payload/response shape.
- It builds a canonical request and consumes a canonical response.

**Impact**: trainer-side logic remains stable while backend-specific details move into adapters.

#### C) Startup path split in RolloutManager

- Existing path (SGLang): keep current managed startup behavior.
- vLLM path: managed mode -- Slime creates `VLLMEngine` Ray actors that launch and manage local vLLM server processes, just like SGLang path uses `SGLangEngine`.

**Impact**: vLLM gets the same lifecycle management as SGLang (process startup, health check, shutdown).

#### D) Weight sync: VLLMEngine with same interface as SGLangEngine

Training-side weight updater (`UpdateWeightFromDistributed`) calls engine methods via Ray remote. The core call chain with source locations:

```text
Training side (Megatron actor, UNCHANGED)
  UpdateWeightFromDistributed.update_weights()
    -> engine.pause_generation.remote()                # Ray remote call
        -> VLLMEngine.pause_generation()                # Ray actor method
            -> requests.post("http://localhost:8000/sleep?level=2")
            -> requests.post("http://localhost:8000/wake_up?tags=weights")

    -> engine.flush_cache.remote()
        -> VLLMEngine.flush_cache()                     # no-op, sleep level 2 covers this

    -> engine.init_weights_update_group.remote()
        -> VLLMEngine.init_weights_update_group()
            -> requests.post("http://localhost:8000/collective_rpc",
                 json={"method": "init_weight_update_group",
                       "master_address": ..., "master_port": ...,
                       "rank_offset": ..., "world_size": ...})

    -> dist.broadcast(param, src=0, group=nccl_group)  # training process NCCL broadcast
    -> engine.update_weights_from_distributed.remote()  # tell vLLM "receive these params"
        -> VLLMEngine.update_weights_from_distributed()
            -> requests.post("http://localhost:8000/collective_rpc",
                 json={"method": "update_weight",
                       "name": ..., "dtype_name": ..., "shape": ...})
            -> vLLM worker: NCCL broadcast recv + model.load_weights()

    -> engine.continue_generation.remote()
        -> VLLMEngine.continue_generation()
            -> requests.post("http://localhost:8000/wake_up?tags=kv_cache")


Rollout generation side
  sglang_rollout.generate()
    -> VLLMClient.generate()
        -> httpx.post("http://localhost:8000/v1/completions")  # direct to local vLLM
```

`VLLMEngine` exposes **the same method signatures** as `SGLangEngine`. Endpoint mapping:

| Method (same signature) | SGLangEngine internal | VLLMEngine internal |
|---|---|---|
| `pause_generation()` | `POST /pause_generation` | `POST /sleep?level=2` + `POST /wake_up?tags=weights` |
| `flush_cache()` | `GET /flush_cache` | no-op (sleep level 2 covers this) |
| `init_weights_update_group(...)` | `POST /init_weights_update_group` | `POST /collective_rpc {"method":"init_weight_update_group",...}` |
| `update_weights_from_distributed(...)` | `POST /update_weights_from_distributed` | `POST /collective_rpc {"method":"update_weight",...}` |
| `continue_generation()` | `POST /continue_generation` | `POST /wake_up?tags=kv_cache` |

**Impact**: training-side code (`UpdateWeightFromDistributed`, `train.py`, actor code) requires **zero changes**. Weight sync uses NCCL broadcast (GPU direct transfer), same efficiency as SGLang path.

Source links:
- [train.py#L88](../../../train.py#L88), [actor_group.py#L126](../../../slime/ray/actor_group.py#L126), [actor.py#L532](../../../slime/backends/megatron_utils/actor.py#L532)
- [update_weight_from_distributed.py#L82](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L82), [#L89](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L89), [#L90](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L90), [#L280](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L280), [#L321](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L321), [#L139](../../../slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L139)
- [sglang_engine.py#L415](../../../slime/backends/sglang_utils/sglang_engine.py#L415), [#L296](../../../slime/backends/sglang_utils/sglang_engine.py#L296), [#L373](../../../slime/backends/sglang_utils/sglang_engine.py#L373), [#L398](../../../slime/backends/sglang_utils/sglang_engine.py#L398), [#L420](../../../slime/backends/sglang_utils/sglang_engine.py#L420)
- [sglang_rollout.py#L47](../../../slime/rollout/sglang_rollout.py#L47), [#L72](../../../slime/rollout/sglang_rollout.py#L72), [#L165](../../../slime/rollout/sglang_rollout.py#L165), [#L311](../../../slime/rollout/sglang_rollout.py#L311)
- [rollout.py#L477](../../../slime/ray/rollout.py#L477), [#L1028](../../../slime/ray/rollout.py#L1028), [#L1041](../../../slime/ray/rollout.py#L1041)

#### E) Router decision: no router for vLLM initial phase

- SGLang Model Gateway: only supports SGLang workers, not applicable.
- SlimeRouter: only needed for R3 / radix-tree caching; Qwen2.5-0.5B is not MoE and uses token-in/token-out.
- Single vLLM instance, `VLLMClient` connects directly to local vLLM server port.

### 4.2 Interface changes

#### A) New CLI interfaces

- `--rollout-backend {sglang,vllm}`
- `--vllm-base-url`
- `--vllm-api-mode` (e.g. OpenAI-compatible completion mode)
- `--vllm-model`
- `--vllm-max-retries`

#### B) New internal canonical interfaces

Add canonical rollout contract types:

- `RolloutBackendRequest`
- `RolloutBackendResponse`

These carry backend-neutral fields such as:

- input token ids
- sampling params
- output token ids/logprobs
- canonical finish reason (`stop|length|abort`)
- backend raw response for debugging

#### C) Capability-gated interface behavior

Backends declare capabilities (abort support, routed experts support, prompt logprobs support). Unsupported features are explicitly gated, logged, and/or failed fast.

## 5. File-level Change Map (for maintainers)

### New files

- `slime/rollout/backends/base_client.py` -- backend client interface + capability model
- `slime/rollout/backends/sglang_client.py` -- extracted SGLang rollout client
- `slime/rollout/backends/vllm_client.py` -- vLLM rollout client
- `slime/rollout/backends/__init__.py` -- adapter exports
- `slime/backends/vllm_utils/vllm_engine.py` -- `VLLMEngine` Ray actor (analogous to `SGLangEngine`)
- `slime/backends/vllm_utils/worker_extension.py` -- vLLM WorkerExtension for NCCL weight sync

### Modified files

- `slime/rollout/base_types.py` -- add canonical backend request/response types
- `slime/rollout/sglang_rollout.py` -- use backend adapters instead of hardcoded SGLang protocol
- `slime/utils/arguments.py` -- add `--rollout-backend`, vLLM args; skip SGLang parse when vLLM
- `slime/ray/rollout.py` -- startup flow split: create `VLLMEngine` when `rollout_backend=vllm`

### Unchanged files (by design)

- `train.py` -- training loop does not know about backend
- `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` -- engine method signatures are identical
- `slime/backends/megatron_utils/actor.py` -- calls engine methods polymorphically

## 6. Compatibility Matrix (initial)

| Capability | SGLang | vLLM (initial) | Handling |
|---|---|---|---|
| Token-level response logprobs | Yes | Partial/endpoint-dependent | Adapter normalization |
| Finish reason mapping (`stop|length|abort`) | Native | Different enums likely | Canonical mapping |
| Worker-level abort | Yes | Typically no direct equivalent | timeout/cancel fallback strategy |
| Prompt logprobs (OPD-related) | Yes | Partial/unknown by endpoint | capability-gated |
| Routed experts replay (R3) | Yes | No | explicit unsupported gate |
| SGLang worker API dependency | Yes | No | isolate in SGLang adapter |

## 7. Phased Plan

### Phase 1: Managed vLLM GRPO training (small-scale validation)

**Goal**: Qwen2.5-0.5B GRPO 8-GPU sync training on GSM8K, loss/reward convergence comparable to SGLang.

Key deliverables:
- `VLLMEngine` Ray actor (process lifecycle, sleep/wake_up, weight sync via NCCL)
- `VLLMClient` rollout adapter (request/response mapping, logprob/finish_reason normalization)
- `RolloutBackendRequest/Response` canonical contract
- `SGLangClient` extraction (isolate SGLang protocol details)
- `start_rollout_servers` branching for vLLM path
- Argument parsing split (`--rollout-backend vllm`)

Technical decisions:
- No router (single vLLM instance, direct connection)
- Weight sync: NCCL broadcast via `VLLMEngine` with same method signatures as `SGLangEngine`
- Non-colocate mode (separate training and rollout GPUs)
- Rollout function: reuse `sglang_rollout.py` (backend-agnostic orchestration)
- Training-side code: zero changes

Acceptance criteria:
- `num_rollout=3` smoke passes without errors
- Weight sync correctness: rollout output changes with each training update
- reward mean delta < 5% vs SGLang path (same seed/config, 20 steps)
- Existing SGLang tests remain green

### Phase 2: Performance, scale, and advanced features

- Colocate mode support (GPU IPC weight transfer)
- Multi-instance vLLM with load balancing
- Abort/cancel strategy refinement
- Prompt logprob support (OPD scenarios)
- Deterministic computation verification
- Larger model validation (e.g. Qwen3-4B)

## 8. Validation strategy

### Unit

- finish reason normalization
- token/logprob alignment
- capability-gated behavior

### Integration

- SGLang vs vLLM response schema comparisons
- timeout/retry/non-crash behavior checks

### End-to-end GRPO

- smoke (short rollout)
- stability (medium horizon)
- stress (higher concurrency / latency pressure)

## 9. Risks and mitigations

- **Logprob semantic mismatch**: strict adapter checks + canonicalization + tests.
- **Abort mismatch**: capability model + timeout/cancel fallback + explicit logs.
- **Training drift**: mandatory A/B runs with fixed seeds and aligned configs.
- **Middleware assumptions**: enforce capability checks and default-disable incompatible middleware paths.

## 10. Open questions for Slime maintainers

1. Should vLLM initial path be direct endpoint only, or standardize via SlimeRouter generic mode immediately?
2. What minimum logprob fidelity is required to claim GRPO support?
3. Should OPD prompt-logprob paths be blocked for vLLM until full parity?
4. What backend quality gates are required before default recommendation?

## 11. Decision requested

Approve incremental implementation with:

- contract-first adapter architecture,
- external vLLM initial support,
- explicit capability-gated behavior,
- strict SGLang non-regression requirement.
