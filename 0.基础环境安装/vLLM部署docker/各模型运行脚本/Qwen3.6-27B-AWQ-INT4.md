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
# #########################################################################################
优化上下文为最大：
好，所有数据都拿到了。有个重要修正——你当前 vLLM 0.19 的 --kv-cache-dtype 不支持 int8，只支持 fp8。RTX 4090（SM 8.9）原生支持 FP8。

  下面按你的启动命令中的参数 + 推荐新增参数，逐一整理。

  ---
  一、当前命令中的参数

  1. 模型加载

  参数: --model
  说明: HuggingFace 模型 ID 或本地路径
  默认值: 无（必填）
  你的当前值: /data/models/Qwen3.6-27B-AWQ-INT4
  推荐值: 不变
  注意事项: 目录需包含 config.json、tokenizer、权重文件
  ────────────────────────────────────────
  参数: --served-model-name
  说明: API 中返回的模型名，客户端 model 字段需匹配此值
  默认值: 使用 --model 的值
  你的当前值: chaos3.6-27b
  推荐值: 不变
  注意事项: 可用逗号分隔设置多个别名
  ────────────────────────────────────────
  参数: --dtype
  说明: 模型计算精度
  默认值: auto（从 config 推断）
  你的当前值: float16
  推荐值: auto
  注意事项: AWQ 反量化内核用 FP16，auto 可正确推断；float16 也行但 auto 更通用
  ────────────────────────────────────────
  参数: --trust-remote-code
  说明: 允许加载 HuggingFace 自定义模型代码
  默认值: False
  你的当前值: ✅ 启用
  推荐值: ✅ 保持启用
  注意事项: Qwen 模型可能需要此标志；有安全风险，仅对可信模型启用
  ────────────────────────────────────────
  参数: --quantization / -q
  说明: 量化方法
  默认值: None（自动检测）
  你的当前值: 未指定（自动检测 AWQ）
  推荐值: 不变
  注意事项: AWQ 模型自带 quantization_config，无需显式指定

  2. 并行

  参数: --tensor-parallel-size / -tp
  说明: 张量并行度，模型按层切分到多 GPU
  默认值: 1
  你的当前值: 2
  推荐值: 2
  注意事项: 27B-AWQ 约 14GB 权重，单卡 24GB 理论可放但 KV Cache 空间不足，TP=2 更安全。总 GPU 数 = tp × pp × dp
  ────────────────────────────────────────
  参数: --pipeline-parallel-size / -pp
  说明: 流水线并行度，按层切分到不同 GPU
  默认值: 1
  你的当前值: 未指定
  推荐值: 1
  注意事项: TP 已足够；PP 引入气泡效率更低；Qwen3.5 混合注意力 PP 已知有问题 (https://github.com/vllm-project/vllm/issues/36643)
  ────────────────────────────────────────
  参数: --data-parallel-size / -dp
  说明: 数据并行度，每 rank 持完整模型副本
  默认值: 1
  你的当前值: 未指定
  推荐值: 1
  注意事项: DP=2 要求单卡能放下完整模型+KV Cache，27B-AWQ 单卡空间不足，故选 TP

  3. 上下文与显存

  参数: --max-model-len
  说明: 模型最大上下文长度（token 数），支持 128k 等写法
  默认值: 从 config 的 max_position_embeddings 推断（262144）
  你的当前值: 131072 (128k)
  推荐值: 262144 (256k)
  注意事项: 直接决定 KV Cache 预分配量，是显存最大消耗项。需搭配 --kv-cache-dtype fp8 才能在 2×4090 上跑到 256k
  ────────────────────────────────────────
  参数: --gpu-memory-utilization
  说明: vLLM 使用的 GPU 显存比例
  默认值: 0.9
  你的当前值: 0.9
  推荐值: 0.95
  注意事项: 0.9 → 0.95 每卡多出 ~1.2GB 给 KV Cache。设太高有 OOM 风险，建议不超过 0.95
  ────────────────────────────────────────
  参数: --max-num-seqs
  说明: 单次迭代最大并发序列数
  默认值: 自动计算（通常 128）
  你的当前值: 8
  推荐值: 8
  注意事项: 你的设置保守但合理——更多并发意味着 KV Cache 被更多序列瓜分，每个序列的可用上下文变短

  4. KV Cache

  参数: --kv-cache-dtype
  说明: KV Cache 数据精度
  默认值: auto（同模型 dtype，即 fp16）
  你的当前值: 未指定（fp16）
  推荐值: fp8
  注意事项: 最关键的新增参数。 KV Cache 内存减半。RTX 4090（SM 8.9）原生支持 FP8。可选值：auto/fp8/fp8_e4m3/fp8_e5m2/float16/bfloat16。注意：不支持 int8，之前我的建议有误
  ────────────────────────────────────────
  参数: --enable-prefix-caching
  说明: 前缀缓存（APC），复用不同请求间的公共 KV Cache 前缀
  默认值: v0.19 默认启用
  你的当前值: ✅ 启用
  推荐值: ✅ 保持启用
  注意事项: 对多轮对话、相同 system prompt 场景效果显著
  ────────────────────────────────────────
  参数: --block-size
  说明: KV Cache 逻辑块大小（token 数）
  默认值: 16
  你的当前值: 未指定
  推荐值: 保持默认
  注意事项: 影响 prefix caching 粒度，通常无需调整

  5. CUDA Graph

  参数: --cudagraph-capture-sizes
  说明: CUDA Graph 捕获的 batch size 列表
  默认值: 自动推断（1 到 max-num-seqs 的 2 的幂次序列）
  你的当前值: 未指定
  推荐值: 1,2,4,8
  注意事项: 限定只捕获你实际需要的 batch size，减少启动时间和显存占用。你的 max-num-seqs=8，只需捕获到 8
  ────────────────────────────────────────
  参数: --enforce-eager
  说明: 禁用 CUDA Graph（强制 eager 模式）
  默认值: False
  你的当前值: 未指定
  推荐值: False
  注意事项: 仅在 CUDA Graph 导致 OOM 或调试时启用。会显著降低 decode 速度

  6. 服务与 API

  参数: --host
  说明: 监听地址
  默认值: 127.0.0.1
  你的当前值: 0.0.0.0
  推荐值: 不变
  注意事项: 允许外部访问
  ────────────────────────────────────────
  参数: --port
  说明: 监听端口
  默认值: 8000
  你的当前值: 8000
  推荐值: 不变
  注意事项: —
  ────────────────────────────────────────
  参数: --enable-auto-tool-choice
  说明: 启用自动工具选择
  默认值: False
  你的当前值: ✅ 启用
  推荐值: ✅ 保持启用
  注意事项: 必须搭配 --tool-call-parser，单独启用无效
  ────────────────────────────────────────
  参数: --tool-call-parser
  说明: Tool call 输出解析器
  默认值: None
  你的当前值: qwen3_coder
  推荐值: 不变
  注意事项: vLLM 0.19 内置 qwen3_coder 解析器。其他可选：hermes、qwen3_xml、pythonic 等（共 30+ 种）
  ────────────────────────────────────────
  参数: --reasoning-parser
  说明: 思维链内容解析器，分离 reasoning 和 content
  默认值: None
  你的当前值: qwen3
  推荐值: 不变
  注意事项: vLLM 0.19 内置 qwen3 解析器。其他可选：deepseek_r1、deepseek_v3、gemma4、mistral 等。若 enable_thinking=false 可不设，但保留无害
  ────────────────────────────────────────
  参数: --default-chat-template-kwargs
  说明: 传递给聊天模板的默认参数
  默认值: None
  你的当前值: {"enable_thinking": false}
  推荐值: 不变
  注意事项: 禁用 Qwen3 思维模式，减少输出 token 消耗，降低延迟。若需要 thinking 模式设为 true 并搭配 --reasoning-parser qwen3

  ---
  二、推荐新增参数

  ┌───────────────────────────┬────────────────────────────────┬────────────────┬──────────┬────────────────────────────────────────────────────────────────────┐
  │           参数            │              说明              │     默认值     │  推荐值  │                                理由                                │
  ├───────────────────────────┼────────────────────────────────┼────────────────┼──────────┼────────────────────────────────────────────────────────────────────┤
  │ --kv-cache-dtype          │ 见上方 KV Cache 部分           │ auto           │ fp8      │ 效果最大。 KV Cache 内存减半，256k 上下文的关键前提                │
  ├───────────────────────────┼────────────────────────────────┼────────────────┼──────────┼────────────────────────────────────────────────────────────────────┤
  │ --cudagraph-capture-sizes │ 见上方 CUDA Graph 部分         │ 自动推断       │ 1,2,4,8  │ 减少 CUDA Graph 预分配显存，加快启动                               │
  ├───────────────────────────┼────────────────────────────────┼────────────────┼──────────┼────────────────────────────────────────────────────────────────────┤
  │ --enable-chunked-prefill  │ 分块预填充，长 prompt 拆分处理 │ v0.19 默认启用 │ 显式启用 │ 长上下文（>4k）场景显著降低 TTFT（首 token 延迟），改善吞吐        │
  ├───────────────────────────┼────────────────────────────────┼────────────────┼──────────┼────────────────────────────────────────────────────────────────────┤
  │ --swap-space              │ 每 GPU 的 CPU swap 空间（GiB） │ 4              │ 4        │ 默认值即可。KV Cache 溢出时交换到 CPU 内存，增加可用容量但增加延迟 │
  └───────────────────────────┴────────────────────────────────┴────────────────┴──────────┴────────────────────────────────────────────────────────────────────┘

  ---
  三、其他可用参数（参考）

  参数: --max-num-batched-tokens
  说明: 每次迭代处理的最大 token 总数
  默认值: 自动计算
  适用场景: 控制每个 prefill chunk 大小。较小值降低延迟但减吞吐。长上下文场景保持默认即可
  ────────────────────────────────────────
  参数: --max-seq-len-to-capture
  说明: CUDA Graph 捕获的最大序列长度
  默认值: 8192
  适用场景: 超过此长度的 decode 不走 CUDA Graph。设到与 max-model-len 相同可覆盖全范围，但会大幅增加捕获时间和显存
  ────────────────────────────────────────
  参数: --load-format
  说明: 权重加载格式
  默认值: auto
  适用场景: AWQ 模型用 safetensors 格式，auto 自动识别
  ────────────────────────────────────────
  参数: --seed
  说明: 随机种子
  默认值: 0
  适用场景: 结果可复现时使用
  ────────────────────────────────────────
  参数: --disable-log-stats
  说明: 禁用周期性统计日志
  默认值: False
  适用场景: 减少日志噪音
  ────────────────────────────────────────
  参数: --disable-log-requests
  说明: 禁用每个请求的日志
  默认值: False
  适用场景: 高并发时减少日志量
  ────────────────────────────────────────
  参数: --optimization-level
  说明: 整体优化级别（0/1/2）
  默认值: 2
  适用场景: 0 = 禁用所有优化；2 = CUDA Graph + kernel 融合等全开
  ────────────────────────────────────────
  参数: --performance-mode
  说明: 吞吐/延迟权衡
  默认值: balanced
  适用场景: 可选 throughput/balanced/latency
  ────────────────────────────────────────
  参数: --compilation-config / -cc
  说明: torch.compile 编译配置
  默认值: 自动
  适用场景: 可进一步优化 kernel 融合，但增加启动时间
  ────────────────────────────────────────
  参数: --disable-custom-all-reduce
  说明: 禁用自定义 all-reduce，回退 NCCL
  默认值: False
  适用场景: TP 通信优化，保持默认。若遇到多卡通信问题可禁用排查
  ────────────────────────────────────────
  参数: --api-key
  说明: API 访问密钥
  默认值: None
  适用场景: 生产环境加认证
  ────────────────────────────────────────
  参数: --chat-template
  说明: 自定义聊天模板路径
  默认值: 使用模型自带
  默认值: 使用模型自带
  默认值: 使用模型自带
  适用场景: Qwen3 模型自带正确模板，通常无需覆盖
  ────────────────────────────────────────
  参数: --hf-overrides
  说明: 覆盖 HuggingFace config 字段
  默认值: None
  适用场景: 如 --hf-overrides '{"num_hidden_layers": 32}' 调试用

  ---
  四、修正后的优化启动命令

nohup python -m vllm.entrypoints.openai.api_server \
        --model /data/models/Qwen3.6-27B-AWQ-INT4 \
        --served-model-name chaos3.6-27b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.95 \
        --max-model-len 262144 \
        --max-num-seqs 8 \
        --dtype auto \
        --kv-cache-dtype fp8 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --cudagraph-capture-sizes 1 2 4 8 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --trust-remote-code \
        > /root/vllm_chaos36_27b_awq.log 2>&1 &

  与你原命令的差异汇总

  ┌───────────────────────────┬─────────────────┬───────────────────────────────────────────┐
  │           变更            │   原值 → 新值   │                   效果                    │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --gpu-memory-utilization  │ 0.9 → 0.95      │ 每卡多 ~1.2GB 给 KV Cache                 │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --max-model-len           │ 131072 → 262144 │ 上下文 128k → 256k（模型原生极限）        │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --dtype                   │ float16 → auto  │ 让 vLLM 自动选择最优精度                  │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --kv-cache-dtype          │ (无) → fp8      │ KV Cache 内存减半，256k 的关键前提        │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --cudagraph-capture-sizes │ (无) → 1,2,4,8  │ 限定 CUDA Graph batch sizes，减少显存浪费 │
  ├───────────────────────────┼─────────────────┼───────────────────────────────────────────┤
  │ --enable-chunked-prefill  │ (无) → 显式启用 │ 长上下文场景降低 TTFT                     │
  └───────────────────────────┴─────────────────┴───────────────────────────────────────────┘
