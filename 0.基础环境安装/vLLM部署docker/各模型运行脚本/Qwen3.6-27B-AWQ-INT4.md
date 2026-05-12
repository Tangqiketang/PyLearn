# 下载
nohup huggingface-cli download cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --local-dir /data/models/Qwen3.6-27B-AWQ-INT4 \
  --local-dir-use-symlinks False \
  > /root/download_qwen36_27b_awq_int4.log 2>&1 &

## 不开启思考
nohup python -m vllm.entrypoints.openai.api_server \
      --model /data/models/Qwen3.6-27B-AWQ-INT4 \
      --served-model-name chaos3.6-27b \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.9 \
      --max-model-len 131072 \
      --max-num-seqs 8 \
      --dtype float16 \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --trust-remote-code \
      > /root/vllm_chaos36_27b_awq.log 2>&1 &

## 开启思考
	nohup python -m vllm.entrypoints.openai.api_server \
      --model /data/models/Qwen3.6-27B-AWQ-INT4 \
      --served-model-name chaos3.6-27b \
      --host 0.0.0.0 \
      --port 8000 \
      --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.9 \
      --max-model-len 131072 \
      --max-num-seqs 8 \
      --dtype float16 \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --trust-remote-code \
      > /root/vllm_chaos36_27b_awq.log 2>&1 &