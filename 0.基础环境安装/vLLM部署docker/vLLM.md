# 1.通用环境准备
120GB内存 30核心 2*4090

## 1.1 Conda 配置国内源
# 配置清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
验证：
cat ~/.condarc

## 1.2 创建虚拟环境
conda create -n vllm python=3.10 -y
conda activate vllm

## 1.3 pip 配置国内源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
验证：
pip config list

## 1.4 确认 CUDA 环境
nvidia-smi            # 确认驱动和 CUDA 版本
nvcc --version        # 确认 CUDA Toolkit 版本
要求：驱动 ≥ 535，CUDA ≥ 12.1。云 GPU 容器实例一般已预装。


=========================================================================
========================================================================

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 并发8，长度32768
nohup python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Qwen3.5-27B-GPTQ-Int4 \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
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

=====================================================================
ai:
  llm:
    # === 私有化部署（vLLM）===
    api-url: http://183.222.230.10:40036/v1/chat/completions
    api-key: EMPTY
    model-name: qwen3.5-27b
    max-tokens: 2048
    temperature: 0.6
    timeout-seconds: 60
    max-agent-rounds: 10
    catalog-cache-ttl-minutes: 10
    max-history-messages: 10