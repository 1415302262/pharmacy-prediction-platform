# Step 08 Hugging Face Platform

## 时间

2026-03-13

## 本步目标

把项目包装成一个可公开访问的交互网页平台，而不只是离线代码。

## 技术选型

- Flask
- Docker
- Hugging Face Spaces

## 功能

- 单分子预测
- JSON API
- 项目核心结果图展示
- 严格 scaffold 验证结果展示
- SHAP 可解释性展示

## 结果

平台目录已建立：`hf_space/`

核心文件：

- `hf_space/app.py`
- `hf_space/templates/index.html`
- `hf_space/static/style.css`
- `hf_space/Dockerfile`
- `hf_space/README.md`

## 结论

平台化让项目从“模型实验”升级为“可访问、可展示、可调用的小型科研工具”。
