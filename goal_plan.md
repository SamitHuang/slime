### 阶段一：打通Qwen2.5-0.5B GRPO 8卡同步/异步训练（train.py和train_async.py），GSM8K 数据集，loss/reward 收敛与 SGLang backend 基本一致，且满足确定性计算，多次重复运行Loss曲线完全一致。

First Design and RFC by 03/06 

#### 初步方案：
- 对标SGLang，Slime 在 Ray 内管理 vLLM 的完整生命周期，包括进程拉起、权重同步、推理暂停/恢复
- 暂不使用Router，SGLang Model Gateway仅只支持SGLang Worker，SlimeRouter仅在 R3 / radix-tree caching 时需要，Qwen2.5-0.5B 非 MoE 且用 token-in/token-out
- 单vLLM实例，无router，通过vLLMClient 直连本地 vLLM 进程端口
- 先支持训推不共卡(non-colocate)，权重同步采用NCCL broadcast，对标SGLang update_weights_from_distributed  (默认）
- 再支持和验证colocate，权重同步采用GPU IPC（vLLM update_weights_from_ipc, update_weights_from_tensor），对标SGLang update_weights_from_tensor，以验证Reproductivity。**IPC 依赖vllm 0.17**

#### 风险：
- slime, sglang版本依赖，和vllm 0.16的版本依赖冲突(numpy, torch, transformers, etc)
- slime代码较挫，可靠性差，强依赖preset docker
- 算力


#### Reference

https://thudm.github.io/slime/advanced/reproducibility.html


### 阶段二：接入vllm-project/router，支持多实例vLLM

- vllm router forked from SGLang Model Gateway

### 阶段三：多节点大规模验证，MoE模型，optional：验证MTP Speculative Decoding，FP8 rollout 等高级特性

- Model: Qwen/Qwen3-30B-A3B or GLM4.7
- Parallel: 16卡 or 128卡, Train mixed EP+FSDP, Rollout EP+DP
- Verify more features:
    - Bf16 train, FP8 rollout 
    - MTP Speculative Decoding
