# 环境配置指南

## 虚拟环境设置

使用虚拟环境可以隔离项目依赖，避免污染系统环境。

## Conda虚拟环境（推荐）

### 创建虚拟环境

```bash
# 创建Python 3.9虚拟环境
conda create -n qsar python=3.9

# 激活环境
conda activate qsar
```

### 安装依赖

```bash
# 进入项目目录
cd /public/home/zhw/cptac/projects/experiment/qsar_project

# 安装依赖包
pip install -r requirements.txt
```

## pip虚拟环境

### 使用venv创建虚拟环境

```bash
# 创建虚拟环境
python -m venv qsar_env

# 激活环境（Linux/Mac）
source qsar_env/bin/activate

# 激活环境（Windows）
qsar_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 依赖包说明

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| pandas | >=1.5.0 | 数据处理 |
| scikit-learn | >=1.2.0 | 机器学习算法 |
| matplotlib | >=3.6.0 | 数据可视化 |
| rdkit | >=2022.9.0 | 化学信息学 |
| numpy | >=1.23.0 | 数值计算 |

### RDKit安装注意事项

RDKit可以通过conda轻松安装：

```bash
# 使用conda安装（推荐）
conda install -c conda-forge rdkit

# 或使用pip
pip install rdkit-pypi
```

**如果遇到安装问题：**

1. **Conda安装失败**：尝试添加conda-forge渠道
   ```bash
   conda config --add channels conda-forge
   conda install rdkit
   ```

2. **权限问题**：使用用户安装
   ```bash
   pip install --user rdkit-pypi
   ```

3. **依赖冲突**：创建新的干净环境
   ```bash
   conda create -n qsar python=3.9
   conda activate qsar
   pip install rdkit-pypi
   ```

## Jupyter Notebook配置

### 安装Jupyter

```bash
pip install jupyter

# 或安装JupyterLab（推荐）
pip install jupyterlab
```

### 启动Notebook

```bash
# 在项目目录下
jupyter notebook

# 或使用JupyterLab
jupyterlab
```

### 安装内核（可选）

如果需要将虚拟环境作为Jupyter内核：

```bash
pip install ipykernel
python -m ipykernel install --user --name=qsar --display-name "QSAR"
```

## 验证安装

### 测试脚本

创建 `test_env.py`：

```python
import sys
print(f"Python version: {sys.version}")

import pandas
print(f"pandas version: {pandas.__version__}")

import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

import matplotlib
print(f"matplotlib version: {matplotlib.__version__}")

from rdkit import Chem
print("RDKit imported successfully!")

import numpy
print(f"numpy version: {numpy.__version__}")

print("\nAll dependencies installed correctly!")
```

运行测试：
```bash
python test_env.py
```

## 常见问题

### Q: Conda很慢怎么办？

使用清华镜像源：

```bash
# 添加镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 设置搜索时优先使用镜像
conda config --set channel_priority strict
```

### Q: pip很慢怎么办？

使用国内镜像：

```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: RDKit安装失败？

尝试以下方法：

1. 使用conda安装（最可靠）
2. 使用rdkit-pypi（pip版本）
3. 从源码编译（复杂，不推荐）

### Q: 如何导出环境配置？

```bash
# 导出Conda环境
conda env export > environment.yml

# 导出pip依赖
pip freeze > requirements_full.txt
```

## 项目目录权限

确保在项目目录有写入权限：

```bash
cd /public/home/zhw/cptac/projects/experiment/qsar_project
ls -la  # 检查权限
```

## GPU支持（可选）

本项目使用的算法（随机森林、SVR）不需要GPU。

如果未来使用深度学习，可以配置GPU支持：

```bash
# 安装PyTorch（示例）
pip install torch torchvision torchaudio
```
