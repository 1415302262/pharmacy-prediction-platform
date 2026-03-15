# QSAR项目结构说明

## 项目目录结构

```
qsar_project/
├── data/                    # 数据目录
│   └── sample_data.csv     # SMILES和pIC50数据
├── src/                    # 源代码目录
│   ├── featurization.py   # 特征工程模块
│   ├── train_model.py     # 模型训练模块
│   └── evaluate.py        # 模型评估模块
├── notebooks/              # Jupyter笔记本
│   └── qsar_demo.ipynb    # 交互式演示
├── figures/                # 结果图片目录
├── models/                 # 保存的模型文件
├── docs/                   # 学习文档
├── run.py                 # 主运行脚本
├── requirements.txt       # Python依赖包
└── report.md              # 最终报告

```

## 目录说明

### data/
存放训练数据。CSV格式，包含两列：
- `smiles`: 分子的SMILES字符串表示
- `pIC50`: 活性值（负对数的IC50）

### src/
存放核心Python代码，每个文件负责一个功能模块：
- `featurization.py`: 将SMILES转换为分子指纹
- `train_model.py`: 训练机器学习模型
- `evaluate.py`: 评估模型并生成可视化

### notebooks/
Jupyter笔记本，提供交互式开发和演示。

### figures/
存放生成的可视化结果：
- `rf_predictions.png`: 随机森林预测结果散点图
- `svr_predictions.png`: SVR预测结果散点图
- `feature_importance.png`: 特征重要性柱状图

### models/
存放训练好的模型文件（.pkl格式）。

## 文件作用说明

| 文件 | 作用 |
|------|------|
| run.py | 项目主入口，执行完整流程 |
| requirements.txt | 项目依赖包列表 |
| report.md | 学术风格的实验报告 |

## 使用流程

1. 安装依赖: `pip install -r requirements.txt`
2. 运行项目: `python run.py` 或 在Jupyter中打开notebook
3. 查看结果: 查看 `figures/` 目录下的图片
4. 阅读报告: 阅读 `report.md`
