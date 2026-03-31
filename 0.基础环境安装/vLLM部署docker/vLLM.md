nvidia-smi
测试docker中使用GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
运行
docker run -d \
  --name vllm-glm \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_ENDPOINT=https://hf-mirror.com \
  vllm/vllm-openai:latest \
  --model THUDM/chatglm3-6b \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.9

参数解析：
THUDM/chatglm3-6b  中文最稳之一（GLM）

显存控制,5060推荐0.85-0.92
--gpu-memory-utilization 0.9

==============
浏览器访问：
http://localhost:8000/v1/models
测试调用：
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "THUDM/chatglm3-6b",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'

=====================
5060参数优化
--max-model-len 4096  显存不够
--gpu-memory-utilization 0.85  降低显存
===========================
