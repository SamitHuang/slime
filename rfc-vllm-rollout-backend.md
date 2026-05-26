# RFC: 在 Slime 中支持 vLLM 作为 Rollout Backend

## 1. 概要

本 RFC 提议在 Slime 中增加 **vLLM** 作为一等 rollout backend，同时保持现有 SGLang 行为不变，不影响 GRPO 训练流程。

核心设计思路：

1. 定义 backend 无关的 rollout 请求/响应契约（`RolloutBackendRequest`/`RolloutBackendResponse`）。
2. 通过 backend adapter（`SGLangClient`、`VLLMClient`）隔离协议差异。
3. 对不等价能力（abort、routed experts、prompt logprobs）做显式 capability gating。
4. Managed 模式 —— Slime 在 Ray 内管理 vLLM 进程生命周期，与 SGLang 路径同等级别。
5. 权重同步利用 vLLM 原生 weight transfer API，根据部署模式自动选择后端：
   - **Colocate 模式**：CUDA IPC（`IPCWeightTransferEngine`），GPU 共享内存零拷贝。
   - **Non-colocate 模式**：NCCL broadcast（`NCCLWeightTransferEngine`），GPU 直传。

**当前状态**：Phase 1 已完成并验证通过。Qwen2.5-0.5B + GSM8K + 4 GPU colocate 模式下 GRPO 训练正常运行。

## 2. 为什么需要这个

Slime 当前在多处假设 SGLang 行为：

- rollout 生成响应格式（`meta_info.finish_reason.type`、`output_token_logprobs`、可选 `routed_experts`）
- router 控制面交互（`/workers`、`/list_workers`、`/abort_request`）
- Ray 内 rollout server 启动流程

因此支持 vLLM 不是"换个 URL"，而是需要在数据面和控制面做兼容层。

## 3. 目标与非目标

### 目标

- 新增 `--rollout-backend {sglang,vllm}`
- SGLang 路径默认不变
- 训练侧接口保持稳定
- 支持 GRPO rollout，具有显式兼容性语义和可观测性
- **支持 colocate 模式**（训练与推理共享 GPU，通过 CUDA IPC 同步权重）

### 非目标（当前阶段）

- 所有 SGLang 特性的完全对等
- vLLM 上的 R3 路由回放
- 多 vLLM 实例 + router 负载均衡

## 4. 架构和接口改动

### 4.1 架构改动

#### 端到端架构图

```text
                          +--------------------+
                          |    TrainerLoop     |
                          +---------+----------+
                                    |
              +---------------------+---------------------+
              | (generate)                    (update_weights)
              v                                           v
    +--------------------+              +------------------------------------+
    |  RolloutFunction   |              | weight updater (自动选择)            |
    |  (sglang_rollout)  |              |  colocate → UpdateWeightFromTensor  |
    +---------+----------+              |  otherwise → ...FromDistributed     |
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
  | (backend 无关消费)                 |
  +-----------------------------------+
```

#### 组件职责对照

| 组件 | RFC 前 | RFC 后 |
|---|---|---|
| `RolloutFunction` | 包含通用 rollout + SGLang 协议细节 | 只包含通用 rollout 编排 |
| Backend 协议层 | 隐含在 rollout 逻辑里 | 显式的 `RolloutBackendClient` adapter |
| `SGLangClient` | 不存在（逻辑分散） | 拥有 SGLang 请求/响应/控制面的全部细节 |
| `VLLMClient` | 不存在 | 拥有 vLLM `/v1/completions` 请求/响应映射 |
| 训练/sample 管线 | 间接消费 SGLang 格式字段 | 消费统一契约字段，backend 无关 |
| 权重同步 | 仅 SGLang IPC 或 NCCL | 自动适配：SGLang IPC / vLLM IPC / vLLM NCCL |

#### 控制面行为拆分

| 控制面行为 | SGLang 路径 | vLLM 路径 |
|---|---|---|
| Worker 注册/发现 API | 支持 | 不需要 |
| Worker 级 abort | 支持（`/abort_request`） | 降级为超时/取消策略 |
| Routed experts 回放 | 支持 | 显式不支持（capability-gated） |
| 健康检查 | 现有 SGLang/SlimeRouter | `GET /health` 直连 |
| 内存管理 (sleep/wake) | `release/resume_memory_occupation` | `POST /sleep` / `POST /wake_up` |

#### A) Rollout backend 抽象层

- `RolloutBackendClient`（`slime/rollout/backends/base_client.py`）：定义 `generate()` + `capabilities` 抽象接口
- `BackendCapabilities` dataclass：声明 `supports_abort`、`supports_routed_experts`、`supports_prompt_logprobs`
- `SGLangClient`（从现有代码提取）、`VLLMClient`（新建）

**影响**：rollout 逻辑调用统一接口，不再直接嵌入 SGLang HTTP 语义。

#### B) Rollout 执行路径

- `sglang_rollout.generate` 构造 `RolloutBackendRequest`，消费 `RolloutBackendResponse`
- 根据 `--rollout-backend` 选择 `SGLangClient` 或 `VLLMClient`
- `VLLMClient` 调用 vLLM OpenAI-compatible `/v1/completions` 端点

**影响**：训练侧逻辑保持稳定，backend 细节移入 adapter。

#### C) RolloutManager 启动路径

- 现有路径（SGLang）：保持不变
- vLLM 路径：`_start_vllm_rollout_servers()` 创建 `VLLMEngine` Ray actor，启动管理本地 vLLM server 进程

**影响**：vLLM 获得与 SGLang 相同的生命周期管理。

#### D) 权重同步

权重同步支持两种模式，根据部署方式自动选择：

**模式选择逻辑**（`actor.py`）：
```python
update_weight_cls = UpdateWeightFromTensor if args.colocate else UpdateWeightFromDistributed
```
与 SGLang 使用完全相同的选择逻辑，backend 无关。

##### D.1) Colocate 模式：CUDA IPC

vLLM server 启动时配置 `--weight-transfer-config '{"backend": "ipc"}'`。

调用链：
```text
UpdateWeightFromTensor.update_weights()
  → engine.pause_generation.remote()           # VLLMEngine → POST /pause?mode=abort
  → _send_to_colocated_vllm_engine()
      → 每个训练 rank:
          reduce_tensor(tensor)                 # 创建 CUDA IPC handle
          {gpu_uuid: ipc_handle}                # 以物理 GPU UUID 为 key
      → dist.gather_object (Gloo)              # 收集所有 TP rank 的 handles
      → pickle + base64 编码
      → engine.update_weights_from_tensor.remote()
          → VLLMEngine → POST /update_weights
            { "update_info": {
                "names": [...],
                "dtype_names": [...],
                "shapes": [...],
                "ipc_handles_pickled": "base64..."
            }}
          → vLLM IPCWeightTransferEngine.receive_weights()
            → 每个 TP worker 根据自己的 GPU UUID 查找 IPC handle
            → func(*args) 重建 tensor → load_weights()
  → engine.continue_generation.remote()        # VLLMEngine → POST /resume
```

关键点：
- 训练进程和 vLLM worker 共享同一物理 GPU，CUDA IPC 实现零拷贝权重传输
- TP>1 时，各 training rank 的 IPC handles 通过 Gloo gather 合并，每个参数包含所有 GPU UUID 的映射
- 训练侧必须保持 tensor 引用直到 `ray.get()` 返回（vLLM 读取完毕）

##### D.2) Non-colocate 模式：NCCL broadcast

vLLM server 启动时配置 `--weight-transfer-config '{"backend": "nccl"}'`。

调用链：
```text
UpdateWeightFromDistributed.update_weights()
  → engine.init_weights_update_group.remote()  # VLLMEngine → POST /init_weight_transfer_engine
      → vLLM NCCLWeightTransferEngine 初始化 StatelessProcessGroup + PyNcclCommunicator
  → 训练侧: NCCLWeightTransferEngine.trainer_init()
      → 与 vLLM 建立匹配的 NCCL 通信组
  → PyNcclCommunicator.broadcast(tensor, src=0)  # GPU 直传到 vLLM worker
  → engine.update_weights_from_distributed.remote()
      → VLLMEngine → POST /update_weights
        { "update_info": { "names": [...], "dtype_names": [...], "shapes": [...] }}
      → vLLM NCCLWeightTransferEngine.receive_weights()
```

关键点：
- 训练侧使用 vLLM 的 `NCCLWeightTransferEngine.trainer_init()` 初始化 NCCL，与 vLLM 内部的 `StatelessProcessGroup` + `PyNcclCommunicator` 兼容
- 不能使用 `torch.distributed.init_process_group`（vLLM 不兼容）

#### VLLMEngine 端点映射

`VLLMEngine` 暴露与 `SGLangEngine` 兼容的方法签名。内部 HTTP 端点映射：

| 方法 | VLLMEngine 内部 HTTP 调用 |
|---|---|
| `pause_generation()` | `POST /pause?mode=abort` |
| `flush_cache()` | no-op |
| `continue_generation()` | `POST /resume` |
| `init_weights_update_group(...)` | `POST /init_weight_transfer_engine` |
| `update_weights_from_distributed(...)` | `POST /update_weights` (NCCL 模式) |
| `update_weights_from_tensor(...)` | `POST /update_weights` (IPC 模式) |
| `release_memory_occupation()` | `POST /sleep?level=1&mode=abort` |
| `resume_memory_occupation()` | `POST /wake_up` |
| `health_generate()` | `GET /health` |
| `shutdown()` | `process.terminate()` |

**影响**：训练侧代码（`UpdateWeightFromTensor`、`UpdateWeightFromDistributed`、actor 代码）使用统一接口，不感知具体 backend。

#### E) Router 决策

vLLM 当前阶段不使用 router：
- 单 vLLM 实例，`VLLMClient` 直连本地 vLLM server 端口
- SGLang Model Gateway 只支持 SGLang worker，不适用
- SlimeRouter 仅在 R3 / radix-tree caching 时需要

### 4.2 接口改动

#### A) 新增 CLI 接口

- `--rollout-backend {sglang,vllm}` —— 选择 rollout backend
- `--vllm-base-url` —— 手动指定 vLLM server 地址（自动管理时无需设置）
- `--vllm-model` —— vLLM 加载的模型路径（默认同 `--hf-checkpoint`）
- `--vllm-max-retries` —— 生成请求最大重试次数
- `--vllm-enforce-eager` —— 是否禁用 CUDA graph（默认 True）

#### B) 新增内部统一接口

`slime/rollout/base_types.py` 中新增：

- `RolloutBackendRequest`：input_ids、sampling_params、return_logprob、return_routed_experts、image_data、session_id
- `RolloutBackendResponse`：text、output_token_ids、output_token_logprobs、finish_reason（`stop|length|abort`）、prompt_tokens、completion_tokens、backend_raw、routed_experts

#### C) 能力门控

`BackendCapabilities` dataclass 声明各 backend 支持的能力：

| 能力 | SGLangClient | VLLMClient |
|---|---|---|
| `supports_abort` | True | False |
| `supports_routed_experts` | True | False |
| `supports_prompt_logprobs` | True | False |

不支持的特性在调用时显式 gate、记录日志、或 fail fast。

## 5. 文件级改动清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `slime/rollout/backends/base_client.py` | `RolloutBackendClient` 抽象接口 + `BackendCapabilities` |
| `slime/rollout/backends/sglang_client.py` | SGLang rollout client（从现有代码提取） |
| `slime/rollout/backends/vllm_client.py` | vLLM rollout client（`/v1/completions` 适配） |
| `slime/rollout/backends/__init__.py` | adapter 导出 |
| `slime/backends/vllm_utils/vllm_engine.py` | `VLLMEngine` Ray actor（进程管理 + 权重同步） |
| `run-qwen2.5-0.5B-vllm.sh` | vLLM 验证脚本（Qwen2.5-0.5B + GSM8K + colocate） |

### 修改文件

| 文件 | 说明 |
|---|---|
| `slime/rollout/base_types.py` | 新增 `RolloutBackendRequest` / `RolloutBackendResponse` |
| `slime/rollout/sglang_rollout.py` | 使用 backend adapter 代替硬编码 SGLang 协议 |
| `slime/utils/arguments.py` | 新增 `--rollout-backend`、vLLM 参数；vLLM 时设置 sglang 别名默认值 |
| `slime/ray/rollout.py` | `_start_vllm_rollout_servers()` 启动 VLLMEngine |
| `slime/backends/megatron_utils/actor.py` | 权重更新器选择：`colocate → UpdateWeightFromTensor`（backend 无关） |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 新增 `_send_to_colocated_vllm_engine()`（CUDA IPC 路径） |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | vLLM NCCL 兼容：`NCCLWeightTransferEngine.trainer_init()` + `PyNcclCommunicator.broadcast()` |

### 不变文件

| 文件 | 说明 |
|---|---|
| `train.py` | 训练循环不感知 backend |
| SGLang 相关文件 | SGLang 路径完全不受影响 |

### 未使用（与 RFC 初稿的差异）

- **无 `worker_extension.py`**：vLLM 原生 weight transfer API 完全满足需求，无需自定义 WorkerExtension
- **无 `collective_rpc`**：使用 vLLM 的 `/init_weight_transfer_engine` 和 `/update_weights` 端点

## 6. 能力兼容矩阵

| 能力 | SGLang | vLLM | 处理方式 |
|---|---|---|---|
| Token 级响应 logprobs | 支持 | 支持（`choice.logprobs.token_logprobs`） | adapter 归一化 |
| Output token IDs | `meta_info` | `choice.token_ids` | adapter 归一化 |
| Finish reason | `stop\|length\|abort` | `stop\|length` + 其他 | canonical 映射 |
| Worker 级 abort | 支持 | 不支持 | 超时/取消降级 |
| Prompt logprobs（OPD） | 支持 | 部分 | capability-gated |
| Routed experts（R3） | 支持 | 不支持 | 显式 gate |
| Colocate（GPU 共享） | 支持（FlattenedTensorBucket IPC） | 支持（CUDA IPC handles） | 各自原生 IPC |
| 内存管理 | `release/resume_memory_occupation` | `POST /sleep` / `POST /wake_up` | 接口统一 |
| `include_stop_str_in_output` | `no_stop_trim` | `include_stop_str_in_output` | 参数映射 |

## 7. 分阶段计划

### 第一阶段：Managed vLLM GRPO 训练 ✅ Done

**目标**：Qwen2.5-0.5B GRPO 4 卡 colocate 训练，GSM8K 数据集，训练正常运行。

已完成交付物：
- `VLLMEngine` Ray actor：进程生命周期管理、`/pause`/`/resume`/`/sleep`/`/wake_up`
- `VLLMClient` rollout adapter：`/v1/completions` 请求映射、logprob/finish_reason/token_ids 归一化
- `RolloutBackendRequest`/`Response` 统一契约
- `SGLangClient` 提取
- `_start_vllm_rollout_servers` 启动路径
- 参数解析分流（`--rollout-backend vllm`，sglang 别名默认值）
- **Colocate 模式**：CUDA IPC 权重传输（`_send_to_colocated_vllm_engine` → `IPCWeightTransferEngine`）
- **Non-colocate 模式**：NCCL 权重传输（`NCCLWeightTransferEngine.trainer_init()` → `PyNcclCommunicator.broadcast()`）
- 验证脚本：`run-qwen2.5-0.5B-vllm.sh`

技术决策：
- 不使用 router（单 vLLM 实例，直连）
- Colocate 权重同步：CUDA IPC（训练和 vLLM 共享 GPU 内存，零拷贝）
- Non-colocate 权重同步：vLLM 原生 NCCL（`StatelessProcessGroup` + `PyNcclCommunicator`）
- Rollout 函数：复用 `sglang_rollout.py`（backend 无关编排）
- 训练侧权重更新器选择与 SGLang 一致（`colocate → UpdateWeightFromTensor`）

验收结果：
- ✅ Qwen2.5-0.5B + GSM8K + 4 GPU colocate 训练运行通过
- ✅ 权重同步正确性：每轮 rollout 输出随训练更新变化
- ✅ 现有 SGLang 路径不受影响

### 第二阶段：多 vLLM 实例 + 异步训推

**目标**：支持多 vLLM 实例 rollout，使用 [vllm-project/router](https://github.com/vllm-project/router) 做负载均衡，以异步训推（`train_async.py`）在更大模型上验证。

关键交付物：
- 集成 [vllm-project/router](https://github.com/vllm-project/router) 作为 vLLM 侧负载均衡器
- `start_rollout_servers` 拉起 N 个 `VLLMEngine` actor + 一个 vllm-router 进程
- `VLLMClient` 生成请求从直连改为走 vllm-router
- 验证异步训推（`train_async.py`）在 vLLM backend 下的正确性

验收标准：
- 多实例（如 2-4 个 vLLM worker）rollout 在 20+ rollout step 下稳定
- 异步训推（`train_async.py`）无 hang、无权重同步竞态
- 相比单实例基线有吞吐提升
- 更大模型验证（如 Qwen3-4B，TP=2）

### 第三阶段（未来）：高级特性

- Abort/cancel 策略完善
- Prompt logprob 支持（OPD 场景）
- 确定性计算验证
- 性能基准测试（vLLM vs SGLang 吞吐量对比）

## 8. 验证策略

### 已完成（Phase 1）

- ✅ 端到端 GRPO 训练：`run-qwen2.5-0.5B-vllm.sh`（Qwen2.5-0.5B, GSM8K, 4 GPU, colocate）
- ✅ 权重同步正确性验证
- ✅ SGLang 路径非回归

### 待完成

- 单元测试：finish reason 归一化、token/logprob 对齐、capability-gated 行为
- 集成测试：SGLang vs vLLM 响应 schema 对比
- 压测：更高并发 / 延迟压力

## 9. 实现中遇到的关键问题及解决

### 9.1 vLLM NCCL 不兼容

**问题**：vLLM 内部使用 `StatelessProcessGroup` + `PyNcclCommunicator` 管理 NCCL 通信，与 `torch.distributed.init_process_group` 不兼容。

**解决**：训练侧使用 `NCCLWeightTransferEngine.trainer_init()` 初始化 NCCL，确保与 vLLM 端建立匹配的通信组。

### 9.2 vLLM logprobs 格式差异

**问题**：vLLM `/v1/completions` 的 logprobs 格式（`choice.token_ids`、`choice.logprobs.token_logprobs`）与 SGLang 不同。

**解决**：`VLLMClient` 中做显式映射，将 vLLM 格式转换为 `RolloutBackendResponse` 统一格式。对于缺失 `token_ids` 的情况，回退到 tokenizer。

### 9.3 Colocate 模式下的 IPC 格式差异

**问题**：SGLang 使用 `FlattenedTensorBucket` + `MultiprocessingSerializer` 做 IPC，vLLM 使用独立的 CUDA IPC handles（`reduce_tensor`）。两者格式不兼容。

**解决**：新增 `_send_to_colocated_vllm_engine()` 函数，直接使用 `torch.multiprocessing.reductions.reduce_tensor` 创建 CUDA IPC handles，以 GPU UUID 为 key。TP>1 时通过 Gloo gather 合并各 rank 的 handles，然后 pickle + base64 编码后通过 `VLLMEngine` 转发到 vLLM 的 `/update_weights` 端点。

### 9.4 缺失的 sglang 参数别名

**问题**：vLLM backend 跳过 `sglang_validate_args()`，导致 `sglang_dp_size` 等别名未设置，后续代码 `AttributeError`。

**解决**：在 `arguments.py` 中为 vLLM backend 添加条件分支，设置 `sglang_dp_size`、`sglang_pp_size`、`sglang_ep_size`、`sglang_tp_size` 等默认值。

## 10. 风险与缓解

- **Logprob 语义不匹配**：严格 adapter 检查 + 归一化 + 测试
- **Abort 不匹配**：能力模型 + 超时/取消降级 + 显式日志
- **训练漂移**：强制 A/B 对照运行，固定 seed 和配置
- **IPC handle 生命周期**：训练侧通过 `kept_alive` 列表 + `ray.get()` 阻塞确保 tensor 在 vLLM 读取完成前不被回收

## 11. 向 Slime 维护者提出的开放问题

1. 是否需要在 Phase 2 中将 vLLM 多实例 + router 作为默认部署模式？
2. 声称支持 GRPO 所需的最低 logprob 保真度是什么？
3. 是否应在完全对等前阻止 vLLM 的 OPD prompt-logprob 路径？
4. 推荐为默认 backend 之前需要满足什么质量门槛？
5. vLLM colocate IPC 路径是否需要支持混合模式（部分 colocate + 部分 distributed）？

## 12. 请求决策

Phase 1 已实现并验证：

- ✅ Contract-first adapter 架构
- ✅ Managed vLLM 模式（与 SGLang 同等生命周期管理）
- ✅ 双模式权重同步（colocate: CUDA IPC / non-colocate: NCCL）
- ✅ 显式 capability-gated 行为
- ✅ 严格 SGLang 非回归

请批准进入 Phase 2：多 vLLM 实例 + router + 异步训推。

