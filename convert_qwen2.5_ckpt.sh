source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
    --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
