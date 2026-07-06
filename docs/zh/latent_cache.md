# Latent Cache（Diffusion 跨请求近似缓存）

Latent cache 用于在新到达的 prompt 和已经生成过的 prompt 足够相似
时**复用上一次推理的中间 latent**，跳过前若干步去噪。TeleFuser 通过外部
**CacheSeek** 包接入该能力：<https://github.com/Tele-AI/CacheSeek>；
旧的仓内 `telefuser/cache_mem/` 实现已移除。

## Latent Cache 与 Feature Cache 的区别

两者解决的问题维度相异：

|      | Feature cache（参见 `feature_cache.md`） | Latent cache（本文档）       |
| ---- | ------------------------------------ | ----------------------- |
| 粒度   | 单次推理内、跨 timestep                     | 跨请求                     |
| 复用键  | step 索引                              | prompt embedding 相似度    |
| 加速目标 | 跳过可近似的 block                         | 跳过整次去噪的前 N 步         |
| 模块   | `telefuser/feature_cache/`           | 外部 `cacheseek` 包       |
| 持久化  | 无（只在请求生命周期内）                         | KV 磁盘/分布式存储 + 向量库 + 元数据 |

单次请求内推理加速用 feature cache；加速**多次**请求推理时
用 latent cache。两者可以同时启用、互不干扰。

---

## Service 接口

TeleFuser 只在解析出需要启用 latent cache 时才会 import CacheSeek。解析结果为启用时，
如果环境中没有安装 CacheSeek，服务启动会直接失败并给出安装提示。配置优先级见
“启动与运行行为”。

启用后，service 层通过 CacheSeek 的 TeleFuser adapter 串联：

```python
adapter.build_query(task_request)
cache_service.lookup(cache_query)
adapter.apply_resume(lookup_result, engine_ctx=task_data)
adapter.on_response(task_request, latent_payload)
cache_service.save(cache_query, outputs)
```

单次请求内的 lookup / resume / save 失败会 best-effort 降级到无缓存路径；
但启动阶段缺包或初始化失败不会被吞掉。

### 在模型 Forward 中的使用

Pipeline 将 service 层注入的 `latent_data` 传到 denoise stage，去噪循环根据
`skip_step` 决定从哪一步开始，同时按 `saved_steps` 对中间 latent 进行保存：

```python
# denoise stage（见 telefuser/pipelines/wan_video/moe_dit_denoising.py）
cached_latent, effective_start_step, saved_steps = parse_latent_data(
    latent_data,
    expected_shape=tuple(latents.shape),
    total_steps=total_steps,
)
if cached_latent is not None:
    latents = cached_latent.to(device=latents.device, dtype=latents.dtype)

saved_steps_set = frozenset(saved_steps)
latent_states_dict: dict[int, torch.Tensor] = {}

for progress_id, timestep in enumerate(timesteps[effective_start_step:]):
    absolute_step = effective_start_step + progress_id
    # snapshot BEFORE scheduler.step：第 k 步存的是进入 step k 的 latent
    if absolute_step in saved_steps_set:
        latent_states_dict[absolute_step] = latents.detach().cpu()
    noise_pred = self.predict_noise_with_cfg(...)
    latents = self.scheduler.step(noise_pred, timesteps[absolute_step], latents)

# pipeline 在最后将 payload 一并返回，供 service 层异步写回缓存
latent_payload = {
    "latent_states_dict": latent_states_dict,
    "saved_steps": saved_steps,
    "final_step": total_steps - 1,
}
return latents, latent_payload
```

`parse_latent_data`（`telefuser/pipelines/wan_video/latent_data_utils.py`）会做
shape / 范围校验，shape 不一致或 `skip_step` 越界时会自动丢弃缓存并降级为
全量去噪，保证主链路不被污染。

---

## 工厂函数

线上路径不再直接构造 `LatentCache`，而是由 CacheSeek 的 TeleFuser 适配器根据
CLI 参数和 pipeline 文件中的 `CACHE_CONFIG` 生成 `(CacheService, TeleFuserCacheAdapter)`：

```python
from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory

cache_service, cache_adapter = CacheServiceFactory.create_cache_service(
    ppl_file="examples/wan_video/wan22_14b_text_to_video_service.py",
    enable_latent_cache=True,
    cache_mode="read_write",  # "read_write" / "read_only" / "write_only"
)
```

TeleFuser 侧只依赖这个 factory 的输入输出契约：

- 传入当前 pipeline 文件路径 `ppl_file`。
- CLI 显式传入的 `enable_latent_cache` / `cache_mode` 会作为覆盖值传给 CacheSeek；
  未传时传 `None`。
- 期望返回 `(cache_service, cache_adapter)`，其中 adapter 提供
  `build_query`、`apply_resume` 和 `on_response`。

需要绕过 TeleFuser service 直接构造缓存服务时，请以 CacheSeek 文档和
`cacheseek/service/config.py` 为准；TeleFuser 不再提供旧的仓内
`LatentCache` 外观类。

---

## Pipeline 配置示例

在 pipeline 文件里声明 `CACHE_CONFIG`：

```python
# examples/wan_video/wan22_14b_text_to_video_service.py
CACHE_CONFIG = dict(
    enable_latent_cache=True,
    latent_cache_dir=os.getenv("TELEFUSER_LATENT_CACHE_DIR", "./latent_cache/wan22_t2v"),
    cache_mode="write_only",
    kv_store_type="local_file",
    vector_store_type="faiss",
    # Qwen3-VL-Embedding-2B hidden_size=2048，必须与 vector_store 维度一致。
    vector_dim=2048,
    key_steps=[5, 10, 15, 20, 25],
    video_embedding_enabled=True,
    video_embedding_model_path=os.getenv("QWEN3VL_EMBEDDING_PATH", ""),
    video_embedding_max_frames=16,
    text_embedding_device_id=1,
    video_embedding_device_id=1,
    video_vector_collection="video",
    rerank_enabled=True,
    rerank_model_path=os.getenv("QWEN3VL_RERANKER_PATH", "/storage/model_zoo/Qwen3-VL-Reranker-2B"),
    rerank_device_id=int(os.getenv("TELEFUSER_RERANK_DEVICE_ID", "0")),
    rerank_top_k=5,
    rerank_score_threshold=0.85,
)
```

下表说明 Wan2.2 service 示例 `CACHE_CONFIG` 中的常用字段和示例默认值。
这里的“默认”指示例文件在没有对应环境变量或 CLI 覆盖时传给 CacheSeek 的值；
CacheSeek 自身的全量字段和内置默认值以 CacheSeek 文档和
`cacheseek/service/config.py` 为准。

| 字段 | 示例默认值 | 说明 |
|---|---|---|
| `enable_latent_cache` | `True` | 示例 pipeline 默认启用 latent cache。 |
| `cache_mode` | `write_only` | 默认只写缓存，适合先预热或生成缓存；可用 `TELEFUSER_CACHE_MODE` 覆盖。 |
| `latent_cache_dir` | `./latent_cache/wan22_t2v` | 缓存根目录；可用 `TELEFUSER_LATENT_CACHE_DIR` 覆盖。 |
| `kv_store_type` | `local_file` | KV 后端类型；可用 `TELEFUSER_KV_STORE_TYPE` 覆盖。 |
| `vector_store_type` | `faiss` | 向量检索后端类型；可用 `TELEFUSER_VECTOR_STORE_TYPE` 覆盖。 |
| `vector_dim` | `2048` | 向量维度，需要与 embedding 模型输出维度一致。 |
| `key_steps` | `[5, 10, 15, 20, 25]` | Pipeline 被要求 snapshot 的 denoise step 列表。 |
| `video_embedding_enabled` | `True` | 启用视频帧 embedding；示例 save 路径会回填 `embedding_video_frames`。 |
| `video_embedding_model_path` | `""` | 视频 embedding 模型路径；可用 `QWEN3VL_EMBEDDING_PATH` 覆盖，空值如何解析由 CacheSeek 决定。 |
| `video_embedding_max_frames` | `16` | 写回缓存前最多采样的视频帧数。 |
| `text_embedding_device_id` / `video_embedding_device_id` | `1` / `1` | embedding 模型使用的逻辑 GPU id；需要按 `CUDA_VISIBLE_DEVICES` 和并行配置调整。 |
| `video_vector_collection` | `video` | 视频向量 collection 名称。 |
| `rerank_enabled` / `rerank_top_k` / `rerank_score_threshold` | `True` / `5` / `0.85` | rerank 示例配置；实际命中策略由 CacheSeek 执行。 |


---

## 使用示例脚本

| Pipeline              | 脚本                                                              | 说明                   |
| --------------------- | --------------------------------------------------------------- | -------------------- |
| Wan2.2 14B T2V（启用缓存）  | `examples/wan_video/wan22_14b_text_to_video_service.py`         | 完整 latent cache 配置示例 |

启动服务：

```bash
# 先把 CacheSeek 安装到 TeleFuser 的同一个 Python 环境中。
# 当前 cacheseek 尚未发布到公开 PyPI，推荐先准备包含 TeleFuser 适配的
# CacheSeek checkout，再从本地路径安装：
# git clone https://github.com/Tele-AI/CacheSeek.git
# git -C CacheSeek checkout <commit-or-branch>
CACHESEEK_REPO=/path/to/CacheSeek
python -m pip install "${CACHESEEK_REPO}"
#
# 如果参与 CacheSeek 开发，需要让源码修改立即生效，可使用 editable 安装：
# python -m pip install -e "${CACHESEEK_REPO}"
#
# 如果对应的 CacheSeek commit 或分支已经推送到 GitHub，也可以直接安装：
# python -m pip install "cacheseek @ git+https://github.com/Tele-AI/CacheSeek.git@<commit-or-branch>"
#
# 未来 cacheseek 发布到当前 pip 包源后，也可以改用：
# python -m pip install cacheseek

python -c "import cacheseek, torch; print(cacheseek.__file__, torch.__version__)"

telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py \
    --port 8000 \
    --cache-mode read_write
```

TeleFuser 不再提供 `cache` extra，也不会自带 CacheSeek 的后端依赖。`cacheseek`
包自身负责声明默认 TeleFuser Wan2.2 路径需要的依赖，例如本地 FAISS 向量后端和
Qwen3-VL embedding / rerank 相关依赖。只有使用 Qdrant 向量后端时才需要安装
`cacheseek[qdrant]`，例如：

```bash
CACHESEEK_REPO=/path/to/CacheSeek
python -m pip install "${CACHESEEK_REPO}[qdrant]"
# 或者从已推送的 GitHub commit / branch 安装：
# python -m pip install "cacheseek[qdrant] @ git+https://github.com/Tele-AI/CacheSeek.git@<commit-or-branch>"
```

如果使用 GitHub 分支安装，需要注意分支是滚动版本；需要可复现实验或生产部署时，
建议使用确认过的 commit。若 `python -m pip install cacheseek` 报包不存在，
说明当前 pip 包源尚未发布该包或未配置到包含该包的私有源。

---

## 启动与运行行为

### Cache mode 三档

| 模式 | 效果 |
|---|---|
| `read_write` | 读取已有缓存；请求完成后也写回新的缓存。 |
| `read_only` | 只读取已有缓存，不写回新的缓存。 |
| `write_only` | 不使用缓存命中结果，只在请求完成后写入缓存。 |

`enable_latent_cache` 和 `cache_mode` 都可以写在 pipeline 文件的 `CACHE_CONFIG`
里，也可以用 CLI 覆盖。优先级是 CLI > `CACHE_CONFIG` > CacheSeek 默认值。
`--cache-mode` 只接受上表三种值；未传 `--cache-mode` 时，TeleFuser 不覆盖
`CACHE_CONFIG` 中的配置。Wan2.2 service 示例的 `cache_mode` 默认值是 `write_only`。

### 启动和失败语义

- 未传 cache CLI 且 pipeline `CACHE_CONFIG` 未启用 latent cache 时，不会加载 CacheSeek。
- pipeline `CACHE_CONFIG` 启用 latent cache 或传入 `--enable-latent-cache` 后，如果
  CacheSeek 未安装或初始化失败，服务启动直接失败。
- 传入 `--disable-latent-cache` 时，即使 pipeline `CACHE_CONFIG` 启用缓存也不会初始化 CacheSeek。
- 单次请求中的 `build_query` / `lookup` / `apply_resume` / `save` 失败会记录 warning，
  并按无缓存路径继续。
