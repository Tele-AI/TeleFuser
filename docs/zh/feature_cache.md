# 特征缓存

特征缓存是一种通过缓存中间特征并在时间步之间重用它们来加速扩散模型推理的技术。TeleFuser 为视频生成模型实现了 AdaTaylorCache。

## 基础接口

所有特征缓存实现都遵循 `BaseFeatureCache` 接口：

```python
class BaseFeatureCache(ABC):
    def should_compute(self, is_cond: bool) -> bool:
        """检查是否需要真实计算。自动增加步数计数器。"""
        pass
    
    def update(self, output: Tensor, ori_input: Tensor, is_cond: bool):
        """使用计算结果更新缓存。"""
        pass
    
    def approximate(self, input: Tensor, is_cond: bool) -> Tensor:
        """从缓存获取近似输出。"""
        pass
    
    def reset():
        """重置所有缓存状态。"""
        pass
```

### 在模型 Forward 中的使用

```python
# 在 DiT block forward 中：
if self.feature_cache.should_compute(cond_flag):
    output = self.forward_blocks(x, ...)
    self.feature_cache.update(output, x, cond_flag)
else:
    output = self.feature_cache.approximate(x, cond_flag)
```

---

## 工厂函数

使用 `create_feature_cache` 创建缓存实例：

```python
from telefuser.feature_cache import create_feature_cache

# 无缓存（默认）
cache = create_feature_cache("none")

# AdaTaylorCache 用于推理加速
cache = create_feature_cache(
    "ada_taylor",
    model_type="Wan2.1-T2V-1.3B",
    num_inference_steps=50,
)

# 校准器用于参数收集
cache = create_feature_cache(
    "calibrator",
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="MyModel",
)
```

---

## AdaTaylorCache

AdaTaylorCache（自适应泰勒缓存）是一种特征缓存策略，结合了：
- **自适应跳过逻辑**：根据连续时间步之间的幅值比自适应跳过计算
- **泰勒级数近似**：在近似跳过步骤时使用泰勒展开获得更高阶精度
- **混合回退**：当经过的步数超过阈值时回退到残差重用

当 `n_derivatives=0` 时，AdaTaylorCache 退化为简单残差缓存（仅残差重用，无泰勒展开）。

### AdaTaylorCache 工作原理

1. **跳过决策**：跟踪连续时间步之间的幅值比，跳过时累积误差，当累积误差 < 阈值且连续跳过次数 ≤ K 时跳过
2. **近似计算**：当跳过时，如果经过步数 ≤ 阈值则使用泰勒级数展开；如果经过步数 > 阈值则回退到残差重用

### 缓存参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `K` | int | 最大连续跳过步数 |
| `retention_ratio` | float | 初始步骤始终计算的比例（不跳过） |
| `thresh` | float | 跳过决策的误差阈值 |
| `cond_mag_ratios` | list | 条件路径的幅值比 |
| `uncond_mag_ratios` | list | 无条件路径的幅值比 |

### 使用 AdaTaylorCache

在您的 pipeline 中启用 AdaTaylorCache：

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline

# 创建 pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
# ... 初始化 pipeline ...

# 启用 AdaTaylorCache 运行
video = pipe(
    prompt="一只猫在弹钢琴",
    num_inference_steps=50,
    enable_ada_taylor_cache=True,
    ada_taylor_n_derivatives=1,  # 使用泰勒展开（设置为 0 则仅使用残差）
    model_type="Wan2.1-T2V-1.3B",
    # ... 其他参数 ...
)
```

`model_type` 参数指定要使用的预校准参数。可在 `telefuser/feature_cache/ada_taylor_cache/params/` 中查看可用模型。

---

## 缓存校准

AdaTaylorCache 需要模型特定的校准参数。使用校准器为新模型生成这些参数。

### 何时需要校准

在以下情况下需要运行校准：
- 使用新的模型架构
- 使用不同的推理设置（例如不同的 `num_inference_steps` 或 `sigma_shift`）
- 针对特定的质量/速度权衡进行微调

### 校准流程

校准流程运行一次 pipeline 来收集残差统计信息：

1. **初始化校准器**：使用您的推理配置进行设置
2. **运行 Pipeline**：执行一次推理（校准数据自动收集）
3. **保存参数**：参数自动保存到 JSON 文件

### 运行校准

#### 使用示例脚本

```bash
python examples/wan_video/wan21_1_3b_text_to_video_cache_calibrate.py \
    --model_root /path/to/Wan2.1-T2V-1.3B/ \
    --num_inference_steps 50 \
    --sigma_shift 8.0 \
    --model_name "Wan2.1-T2V-1.3B" \
    --output_path ./my_cache_params.json
```

#### 编程方式使用

```python
from telefuser.feature_cache import create_feature_cache

# 创建校准器
cache = create_feature_cache(
    "calibrator",
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="Wan2.1-T2V-1.3B",
)

# 在模型上设置校准器
pipeline.denoise_stage.dit.set_ada_taylor_cache_calibrator(
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="Wan2.1-T2V-1.3B",
)

# 运行 pipeline（校准自动进行）
video = pipeline(
    prompt="一个示例提示词",
    num_inference_steps=50,
    sigma_shift=8.0,
)
```

### 校准输出

生成的 JSON 文件包含：

```json
{
    "K": 0,
    "retention_ratio": 0.0,
    "thresh": 0.0,
    "sigma_shift": 8.0,
    "num_inference_steps": 50,
    "cond_mag_ratios": [1.0, 1.0124, 1.00166, ...],
    "uncond_mag_ratios": [1.0, 1.02213, 1.0041, ...]
}
```

**重要提示**：`K`、`retention_ratio` 和 `thresh` 默认设置为 0。您需要根据质量/速度需求调整这些值：

- **更高的 `K`**：更激进的跳过，推理更快，可能损失质量
- **更高的 `retention_ratio`**：更多初始步骤被计算，质量更好但速度较慢
- **更高的 `thresh`**：对误差更宽容，推理更快，可能损失质量

### 推荐值

对于 Wan2.1 1.3B 模型，推荐的起始值：

```json
{
    "K": 4,
    "retention_ratio": 0.2,
    "thresh": 0.12
}
```

### 参数文件位置

默认情况下，参数保存到：
```
telefuser/feature_cache/ada_taylor_cache/params/{model_name}.json
```

模型名称会被清理（点和斜杠替换为下划线）作为文件名。

---

## AdaTaylorCache 参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `model_type` | str | 必需 | 模型类型，用于加载缓存参数 |
| `n_derivatives` | int | 1 | 泰勒展开阶数（0 表示仅残差，1-2 推荐） |
| `taylor_threshold` | int | 2 | 切换到残差重用的阈值（经过步数 > 阈值时使用残差） |

以下参数从预校准的参数加载：
- `K`：最大连续跳过步数
- `thresh`：跳过决策的误差阈值
- `retention_ratio`：初始步骤始终计算的比例

### 使用示例脚本

```bash
python examples/wan_video/wan21_1_3b_text_to_video_ada_taylor_cache.py \
    --gpu_num 1 \
    --n_derivatives 1 \
    --taylor_threshold 2 \
    --num_inference_steps 40
```

### 何时使用不同配置

- **`n_derivatives=0`**：简单残差缓存，最快，适合速度关键场景
- **`n_derivatives=1`**：泰勒展开配合混合回退，质量速度最佳平衡
- **`n_derivatives=2`**：高阶泰勒展开，更好精度但内存消耗更大

---

## 可用的预校准模型

| 模型 | 文件 | 默认步数 |
|-------|------|---------------|
| Wan2.1-T2V-1.3B | `Wan2_1-T2V-1_3B.json` | 50 |
| Wan2.1-T2V-14B | `Wan2_1-T2V-14B.json` | 50 |
| Wan2.1-I2V-14B-480P | `Wan2_1-I2V-14B-480P.json` | 50 |
| Wan2.1-I2V-14B-720P | `Wan2_1-I2V-14B-720P.json` | 50 |
| Wan2.1-FL2V-14B-720P | `Wan2_1-FL2V-14B-720P.json` | 50 |
| Wan2.2-T2V-A14B | `Wan2_2-T2V-A14B.json` | 50 |
| Wan2.2-I2V-A14B | `Wan2_2-I2V-A14B.json` | 40 |
| Wan2.2-FL2V-A14B | `Wan2_2-FL2V-A14B.json` | 40 |
| HunyuanVideo-T2V | `HunyuanVideo-T2V.json` | 50 |
| HunyuanVideo-I2V | `HunyuanVideo-I2V.json` | 50 |
| Qwen-Image | `Qwen-Image.json` | 50 |
| Qwen-Image-Edit-Plus | `Qwen-Image-Edit-Plus.json` | 40 |

---

## 校准脚本

| Pipeline | 脚本 | 模型类型 |
|----------|--------|------------|
| Wan2.1 T2V 1.3B | `examples/wan_video/wan21_1_3b_text_to_video_cache_calibrate.py` | Wan2.1-T2V-1.3B |
| Wan2.2 I2V A14B | `examples/wan_video/wan22_14b_image_to_video_cache_calibrate.py` | Wan2.2-I2V-A14B |
| HunyuanVideo T2V | `examples/hunyuan_video/hunyuan_video_t2v_cache_calibrate.py` | HunyuanVideo-T2V |
| HunyuanVideo I2V | `examples/hunyuan_video/hunyuan_video_i2v_cache_calibrate.py` | HunyuanVideo-I2V |
| Qwen-Image T2I | `examples/qwen_image/qwen_image_cache_calibrate.py` | Qwen-Image |
| Qwen-Image Edit | `examples/qwen_image/qwen_image_edit_plus_cache_calibrate.py` | Qwen-Image-Edit-Plus |

**Wan2.2 I2V 注意事项：** Wan2.2 使用双分支架构（dit_high + dit_low）。校准脚本在两个分支之间共享一个校准器，以便在单个 JSON 文件中捕获完整的去噪过程。

---

## CFG 并行（CFGP）支持

AdaTaylorCache 支持 CFG 并行模式，每个 rank 处理一个 CFG 分支（cond 或 uncond）。每个路径在内部维护自己的步数计数器，确保 rank 之间正确同步。

```python
# 在使用 2 GPU 的 CFGP 模式下：
# Rank 0：处理 cond 路径，cond_state 步数计数器增加
# Rank 1：处理 uncond 路径，uncond_state 步数计数器增加
# 两个计数器自动保持同步
```

---

## 参考文献

AdaTaylorCache 受到以下工作的启发并建立在其基础之上：

- **MagCache**: Ma, X., Fang, G., Wang, X., et al. (2025). "Semantically-aware Taylor Expansion for Diffusion Model Sampling Acceleration." arXiv preprint arXiv:2506.09045. [链接](https://arxiv.org/abs/2506.09045)

- **TaylorSeer**: Ma, X., Fang, G., Wang, X., et al. (2025). "From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeer." arXiv preprint arXiv:2503.06923. [链接](https://arxiv.org/abs/2503.06923)

我们感谢原作者在通过特征缓存和泰勒级数近似加速扩散模型方面的开创性工作。