# 下载：
nohup huggingface-cli download Qwen/Qwen3.5-27B-GPTQ-Int4 \
  --local-dir /data/models/Qwen3.5-27B-GPTQ-Int4 \
  --local-dir-use-symlinks False \
  > /root/download_qwen3.5_27b.log 2>&1 &


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 

# 运行：
nohup python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen3.5-27B-GPTQ-Int4 \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 98304 \
    --max-num-seqs 8 \
    --dtype float16 \
    --quantization gptq_marlin \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code \
    > /root/vllm_qwen35_27b.log 2>&1 &