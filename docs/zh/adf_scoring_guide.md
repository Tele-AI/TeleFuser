# ADF (Agent Dev First) 项目评分标准

本文档定义了Agent Dev First项目的量化评分标准，用于评估项目对AI Agent的友好程度。

## 一、评分维度

ADF评分包含五个维度，总分50分：

| 维度 | 权重 | 满分 | 核心目标 |
|------|------|------|----------|
| 1. 架构清晰性 | 20% | 10分 | 依赖可被Agent快速理解 |
| 2. 配置统一性 | 20% | 10分 | 配置来源单一、类型安全 |
| 3. 模块化边界 | 20% | 10分 | Agent可在安全边界内操作 |
| 4. 测试可观测性 | 20% | 10分 | Agent可获得明确反馈 |
| 5. 模式可复制性 | 20% | 10分 | Agent可复制参考实现 |

---

## 二、评分细则

### 2.1 架构清晰性（10分）

**定义：** 项目依赖关系可被Agent快速理解，无隐式约定。

| 指标 | 标准 | 分值 | 检测方法 |
|------|------|------|----------|
| 依赖深度 | ≤4层 | 3分 | 依赖图BFS分析 |
| 循环依赖 | 0对 | 4分 | 静态分析检测 |
| 核心模块公开API | ≤5个 | 2分 | 统计`__all__` |
| 目录结构清晰度 | 模块职责从路径可推断 | 1分 | 人工评估 |

**评分规则：**

| 循环依赖数量 | 得分 |
|-------------|------|
| 0对 | 4分 |
| 1-2对（有缓解措施） | 2分 |
| 3-5对 | 1分 |
| >5对 | 0分 |

| 依赖深度 | 得分 |
|----------|------|
| ≤3层 | 3分 |
| 4层 | 2分 |
| 5层 | 1分 |
| >5层 | 0分 |

**排除规则：**
- `utils/`、`test/`、`docs/` 等非逻辑模块不参与循环依赖检测
- 动态导入（`importlib`）不视为编译期依赖

---

### 2.2 配置统一性（10分）

**定义：** 所有可配置参数集中在单一类型安全的数据结构中。

| 指标 | 标准 | 分值 | 检测方法 |
|------|------|------|----------|
| 配置来源数 | 1个 | 3分 | 统计配置加载位置 |
| 类型注解覆盖率 | 100% | 3分 | mypy静态检查 |
| 默认值覆盖率 | 100% | 2分 | dataclass字段分析 |
| 层级组合 | 支持嵌套组合 | 2分 | 代码审查 |

**评分规则：**

| 配置来源数量 | 得分 |
|-------------|------|
| 1个（单一dataclass） | 3分 |
| 2个（dataclass + 环境变量覆盖） | 2分 |
| 3+个（分散在多处） | 0分 |

| 类型注解覆盖率 | 得分 |
|---------------|------|
| 100% | 3分 |
| ≥80% | 2分 |
| ≥60% | 1分 |
| <60% | 0分 |

---

### 2.3 模块化边界（10分）

**定义：** 模块大小、职责、公开接口受严格约束，Agent可在边界内安全操作。

| 指标 | 标准 | 分值 | 检测方法 |
|------|------|------|----------|
| 核心文件行数 | ≤500行 | 3分 | wc -l统计 |
| 超大文件比例 | ≤10% | 2分 | 统计>500行文件占比 |
| 公开API数量/模块 | ≤5个 | 2分 | 统计`__all__` |
| 单元测试隔离率 | 100% | 3分 | 测试fixture独立性检查 |

**评分规则：**

| >1000行文件数量 | 得分 |
|----------------|------|
| 0个 | 3分 |
| 1-3个 | 2分 |
| 4-10个 | 1分 |
| >10个 | 0分 |

| >500行文件占比 | 得分 |
|---------------|------|
| ≤5% | 2分 |
| ≤10% | 1.5分 |
| ≤20% | 1分 |
| >20% | 0分 |

| 公开API>10的模块数 | 得分 |
|-------------------|------|
| 0个 | 2分 |
| 1-3个 | 1分 |
| >3个 | 0分 |

---

### 2.4 测试可观测性（10分）

**定义：** Agent执行任务后可获得明确的成功/失败判定和问题定位。

| 指标 | 标准 | 分值 | 检测方法 |
|------|------|------|----------|
| 单元测试运行时间 | ≤30秒 | 2分 | pytest --durations |
| 错误定位精度 | 文件:行号 | 2分 | 测试输出分析 |
| 测试标记系统 | 有 | 1分 | 检查@pytest.mark |
| Profile工具 | 有 | 2分 | 工具可用性检查 |
| 测试覆盖率 | ≥80% | 2分 | pytest-cov |
| 测试文件数 | ≥50个 | 1分 | 统计test目录 |

**评分规则：**

| 测试覆盖率 | 得分 |
|-----------|------|
| ≥80% | 2分 |
| ≥60% | 1.5分 |
| ≥40% | 1分 |
| <40% | 0分 |

| Profile工具 | 得分 |
|------------|------|
| 专用CLI工具 + 瓶颈分析输出 | 2分 |
| 基础profiler | 1分 |
| 无 | 0分 |

---

### 2.5 模式可复制性（10分）

**定义：** Agent可通过复制参考实现、填充差异完成新功能开发。

| 指标 | 标准 | 分值 | 检测方法 |
|------|------|------|----------|
| 参考实现数量 | ≥1个完整pipeline | 3分 | 统计完整实现 |
| 开发指南步骤数 | ≤5步 | 2分 | 文档分析 |
| Skill系统 | 有 | 3分 | 检查.claude/skills |
| 自动化验证工具 | 有 | 2分 | CI/CD配置检查 |

**评分规则：**

| 参考实现完整性 | 得分 |
|---------------|------|
| 完整可运行的pipeline + 详细注释 | 3分 |
| 完整实现但注释不足 | 2分 |
| 仅有骨架模板 | 1分 |
| 无参考实现 | 0分 |

| Skill系统 | 得分 |
|----------|------|
| ≥3个专用Skills | 3分 |
| 1-2个Skills | 2分 |
| 无Skill但有详细文档 | 1分 |
| 无 | 0分 |

---

## 三、评分等级

| 总分 | 等级 | 描述 | Agent效率 |
|------|------|------|-----------|
| 45-50分 | A+ | 优秀 | Agent可独立完成大部分任务 |
| 40-44分 | A | 良好 | Agent需要少量澄清 |
| 35-39分 | B | 合格 | Agent需要中等程度的引导 |
| 30-34分 | C | 待改进 | Agent需要频繁人工干预 |
| <30分 | D | 不达标 | 不适合Agent开发 |

---

## 四、评分示例

### TeleFuser（2024年3月）

| 维度 | 得分 | 说明 |
|------|------|------|
| 架构清晰性 | 9/10 | 0循环依赖，深度4层 |
| 配置统一性 | 10/10 | 单一dataclass，100%类型安全 |
| 模块化边界 | 8.5/10 | 9个超大文件，其余达标 |
| 测试可观测性 | 9/10 | 93个测试文件，Profile工具基础 |
| 模式可复制性 | 9/10 | 11个参考Pipeline，无Skill系统 |
| **总分** | **45.5/50 (91%)** | **A级** |

### SGLang multimodal_gen（2024年3月）

| 维度 | 得分 | 说明 |
|------|------|------|
| 架构清晰性 | 8/10 | 7对循环依赖，深度4层 |
| 配置统一性 | 7/10 | dataclass + 环境变量 |
| 模块化边界 | 5/10 | 23个超大文件 |
| 测试可观测性 | 8/10 | 27个测试，Profile Skill |
| 模式可复制性 | 9/10 | 6个专用Skills |
| **总分** | **37/50 (74%)** | **B级** |

---

## 五、自动化检测脚本

```python
#!/usr/bin/env python3
"""ADF评分自动化检测脚本"""

import os
import re
import ast
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any

@dataclass
class ADFScore:
    architecture: float  # 架构清晰性
    configuration: float  # 配置统一性
    modularity: float     # 模块化边界
    testing: float        # 测试可观测性
    reproducibility: float  # 模式可复制性

    @property
    def total(self) -> float:
        return self.architecture + self.configuration + self.modularity + self.testing + self.reproducibility

    @property
    def grade(self) -> str:
        if self.total >= 45:
            return "A+"
        elif self.total >= 40:
            return "A"
        elif self.total >= 35:
            return "B"
        elif self.total >= 30:
            return "C"
        else:
            return "D"


def detect_circular_dependencies(project_path: str, exclude: list[str] = None) -> list[tuple[str, str]]:
    """检测循环依赖"""
    if exclude is None:
        exclude = ["utils", "test", "docs", "benchmarks"]

    depends_on = defaultdict(set)
    all_modules = set()

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                module = filepath.replace(project_path, '').split('/')[0].strip('/')
                if module in exclude:
                    continue
                all_modules.add(module)

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()

                # 提取import
                imports = re.findall(r'from ([a-zA-Z_]+)\.', content)
                imports += re.findall(r'import ([a-zA-Z_]+)', content)

                for imp in imports:
                    if imp in all_modules and imp != module:
                        depends_on[module].add(imp)

    # 检测循环
    cycles = []
    checked = set()
    for m1 in all_modules:
        for m2 in all_modules:
            if m1 < m2 and (m1, m2) not in checked:
                if m1 in depends_on.get(m2, set()) and m2 in depends_on.get(m1, set()):
                    cycles.append((m1, m2))
                checked.add((m1, m2))

    return cycles


def count_large_files(project_path: str, threshold: int = 500) -> tuple[int, int, int]:
    """统计超大文件"""
    total = 0
    over_500 = 0
    over_1000 = 0

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = len(file.readlines())
                total += 1
                if lines > 500:
                    over_500 += 1
                if lines > 1000:
                    over_1000 += 1

    return total, over_500, over_1000


def count_public_apis(project_path: str) -> dict[str, int]:
    """统计公开API数量"""
    result = {}

    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        if '__init__.py' in files:
            init_path = os.path.join(root, '__init__.py')
            with open(init_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                items = match.group(1)
                exports = re.findall(r'"([^"]+)"', items) + re.findall(r"'([^']+)'", items)
                module = root.replace(project_path, '').strip('/').replace('/', '.')
                if exports:
                    result[module] = len(exports)

    return result


def score_architecture(project_path: str) -> float:
    """架构清晰性评分"""
    score = 0.0

    # 循环依赖检测
    cycles = detect_circular_dependencies(project_path)
    if len(cycles) == 0:
        score += 4
    elif len(cycles) <= 2:
        score += 2
    elif len(cycles) <= 5:
        score += 1

    # 依赖深度（简化检测）
    score += 2  # 默认给2分，需手动验证

    # 核心模块API
    apis = count_public_apis(project_path)
    core_apis = [v for k, v in apis.items() if 'core' in k]
    if core_apis and max(core_apis) <= 5:
        score += 2
    elif core_apis and max(core_apis) <= 10:
        score += 1

    # 目录结构清晰度
    score += 1  # 需人工评估

    return min(score, 10.0)


def score_modularity(project_path: str) -> float:
    """模块化边界评分"""
    score = 0.0

    total, over_500, over_1000 = count_large_files(project_path)

    # >1000行文件
    if over_1000 == 0:
        score += 3
    elif over_1000 <= 3:
        score += 2
    elif over_1000 <= 10:
        score += 1

    # >500行文件比例
    ratio = over_500 / total if total > 0 else 0
    if ratio <= 0.05:
        score += 2
    elif ratio <= 0.10:
        score += 1.5
    elif ratio <= 0.20:
        score += 1

    # 公开API数量
    apis = count_public_apis(project_path)
    over_10_apis = sum(1 for v in apis.values() if v > 10)
    if over_10_apis == 0:
        score += 2
    elif over_10_apis <= 3:
        score += 1

    # 测试隔离（需人工评估）
    score += 1.5

    return min(score, 10.0)


def evaluate_adf(project_path: str) -> ADFScore:
    """评估项目ADF分数"""

    # 架构清晰性
    architecture = score_architecture(project_path)

    # 配置统一性（需人工评估）
    configuration = 8.0  # 默认值

    # 模块化边界
    modularity = score_modularity(project_path)

    # 测试可观测性（需人工评估）
    testing = 8.0  # 默认值

    # 模式可复制性（需人工评估）
    reproducibility = 8.0  # 默认值

    return ADFScore(
        architecture=architecture,
        configuration=configuration,
        modularity=modularity,
        testing=testing,
        reproducibility=reproducibility
    )


if __name__ == "__main__":
    import sys
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."

    score = evaluate_adf(project_path)

    print(f"ADF评分报告: {project_path}")
    print("=" * 50)
    print(f"架构清晰性:   {score.architecture:.1f}/10")
    print(f"配置统一性:   {score.configuration:.1f}/10")
    print(f"模块化边界:   {score.modularity:.1f}/10")
    print(f"测试可观测性: {score.testing:.1f}/10")
    print(f"模式可复制性: {score.reproducibility:.1f}/10")
    print("-" * 50)
    print(f"总分: {score.total:.1f}/50 ({score.total*2:.0f}%)")
    print(f"等级: {score.grade}")
```

---

## 六、持续改进建议

### 高优先级（+3分潜力）

1. **消除循环依赖** - 重构模块边界，使用依赖注入
2. **拆分超大文件** - 将>1000行文件拆分为多个子模块
3. **统一配置来源** - 移除环境变量配置，使用单一dataclass

### 中优先级（+2分潜力）

4. **增强Profile工具** - 添加CLI入口和瓶颈分析输出
5. **减少公开API** - 合并相似功能，精简接口
6. **补充Skill系统** - 添加常用任务的专用Skill

### 低优先级（+1分潜力）

7. **完善测试隔离** - 确保每个模块有独立测试fixture
8. **增加参考实现** - 覆盖更多使用场景
9. **添加CI检查** - 自动检测行数、循环依赖等

---

## 七、参考资料

- Agent Dev First: 面向AI代理的软件工程范式革新
- [TeleFuser CLAUDE.md](https://github.com/TeleFuser/TeleFuser/blob/main/CLAUDE.md)
- [SGLang Skills Documentation](https://github.com/sgl-project/sglang)