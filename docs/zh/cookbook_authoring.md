# 发布 Cookbook 指南

模型使用正文维护在 `examples/<model>/README.md`，中文译文可放在 `README_zh.md`，较长的部署或性能专题
可在同目录拆分为其他 Markdown 文件。网站构建时生成 Cookbook。公共服务、配置和架构说明继续维护在
`docs/`，示例通过链接引用。

## 登记发布清单

在 `docs/cookbook.yml` 的 `pages` 中显式登记页面：

```yaml
- slug: model-task
  category: world-models
  title:
    en: Model Task
    zh: 模型任务
  sources:
    en: examples/model/README.md
    zh: examples/model/README_zh.md
```

英文来源必填。没有译文时省略 `sources.zh`，中文网址会明确展示英文回退。已有 `README_zh.md` 时必须登记。
`slug` 是固定网址标识，独立于目录名、导航标题和正文标题。示例网址为 `/TeleFuser/cookbook/model-task/`
与 `/TeleFuser/zh/cookbook/model-task/`。分类需要中英文标题；公共双语文档可通过 `guides` 登记，其正文和
原有地址保持不变。未登记的实验示例不会自动发布。

## 正文规范

参考示例 README 模板，按任务、已验证环境、权重与输入、最短运行步骤、预期输出、可选配置、性能和常见问题
组织阅读路径。命令明确从仓库根目录执行。服务示例包含客户端调用；交互示例包含浏览器地址、连接、结果检查
和退出步骤。

文档构建通过不代表模型运行验证通过。性能数据需要硬件、软件版本、日期、代码版本、工作负载和测量范围；
历史记录缺少的信息应明确注明。流式任务分别报告首帧延迟、控制响应、目标端计算速度和客户端交付速度。

## 链接与资源

采用 GitHub 可读的普通 Markdown，支持表格、引用式链接、行内代码和代码块。路径按原始 Markdown 文件目录
解析：已发布示例转为 Cookbook 页面，`docs/` 链接转为已有站内文档，并优先保持当前语言；脚本和未发布文件
转为 GitHub `main` 源文件链接；外部链接、查询参数和锚点保留。跨语言章节链接应在两个译文中使用相同的
显式 HTML 锚点。构建会检查实际渲染出的 ID。缺失文件和绝对仓库路径会导致失败。

支持 HTML 图片、视频、音频和 source 标签的 `src`、`poster` 属性，以及 HTML 链接。请用显式 `src` 代替
`srcset`。允许的本地资源类型为 PNG、JPEG、GIF、WebP、SVG、AVIF、MP4、WebM、MP3、OGG，每个文件上限
5 MiB，总量上限 25 MiB。仅复制引用到的资源，并按完整仓库路径隔离；资源必须提交到 Git。符号链接、权重和
数据集不打包，大型演示媒体使用外部 URL。代码中的路径保持不变。生成 Markdown 可能规范化空白和展开引用
链接，原始正文文件不会被修改。

编辑和源码按钮指向原始文档，英文回退页面也指向英文来源。修改与创建时间从原始文件的 Git 历史读取，并
复用原有日期格式化器；未跟踪文件不显示日期，CI 需要完整历史。不会使用临时生成文件的时间。

## 构建与预览

在仓库根目录执行：

```bash
python -m pip install -r docs/requirements.txt
python -m unittest discover -s tests/docs -v
python scripts/docs/prepare_cookbook.py build
python scripts/docs/prepare_cookbook.py serve --dev-addr 127.0.0.1:8000
```

打开 `http://127.0.0.1:8000/TeleFuser/zh/cookbook/`。修改示例正文后停止预览，再次运行 `serve`；第一版启动时
重新生成，不监听 `examples/` 的实时变更。

`prepare` 只生成临时文档树和继承配置，`check` 校验已经构建的网站。本地与 CI 均使用 `.build/docs/` 和
`.build/mkdocs.yml`，每次生成清除旧页面和未使用资源。不要编辑或提交 `.build/`、`site/`。原有主题、插件、
脚本、样式、专题导航和部署目标保留；继承配置关闭日期插件对临时文件的扫描，由源码 hook 提供日期字段。

CI 覆盖文档、示例、文档脚本及测试的变更，检查清单、运行测试、构建双语页面、校验 Cookbook 链接和渲染后
锚点，并拒绝超出 `scripts/docs/warnings-baseline.txt` 的新增告警。历史告警减少无需更新基线，新增问题必须
修复。工作流提供 PR 检查；仓库分支保护需要将该检查设为必需，才能阻止失败的 PR 合并。
