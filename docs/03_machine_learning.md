# 机器学习模型训练学习文档

## 项目使用的机器学习算法

本项目使用两种回归算法：
1. **随机森林回归 (Random Forest Regressor)**
2. **支持向量回归 (Support Vector Regression, SVR)**

---

## 随机森林回归

### 基本原理

随机森林是一种集成学习方法，由多个决策树组成。每个决策树在训练数据的随机子集上训练，预测时取所有树的平均值。

### 关键概念

#### 决策树 (Decision Tree)

决策树通过一系列"是/否"问题将数据分层，最终到达叶子节点进行预测。

**简单示例：**
```
分子有氯原子吗？
├── 是 → pIC50预测值较高
└── 否 → 分子量大于300吗？
    ├── 是 → pIC50预测值中等
    └── 否 → pIC50预测值较低
```

#### 集成学习 (Ensemble Learning)

"三个臭皮匠，顶个诸葛亮"——组合多个弱学习器形成强学习器。

### 随机森林的随机性

1. **数据随机**: 每棵树使用Bootstrap采样（有放回随机抽样）
2. **特征随机**: 每个分裂点只考虑部分随机特征

### 参数设置

| 参数 | 值 | 说明 |
|------|---|------|
| n_estimators | 100 | 树的数量 |
| random_state | 42 | 随机种子（可重复性） |
| n_jobs | -1 | 使用所有CPU核心（并行计算） |

### 优缺点

**优点：**
- 不易过拟合（通过随机性）
- 无需特征标准化
- 提供特征重要性
- 处理缺失值能力强

**缺点：**
- 模型较大，占用内存
- 预测时相对较慢
- 难以完全解释单棵树

---

## 支持向量回归 (SVR)

### 基本原理

SVR是支持向量机(SVM)的回归版本。它寻找一个超平面，使得大部分数据点落在该平面两侧的边界内。

### 关键概念

#### 超平面 (Hyperplane)

在特征空间中将数据分开或近似数据的平面。

#### 核函数 (Kernel Function)

将数据映射到高维空间，使线性可分/可近似。

**常用核函数：**
- Linear: 适用于线性关系
- RBF (径向基函数): 适用于非线性关系（本项目使用）
- Polynomial: 多项式关系

### 参数设置

| 参数 | 值 | 说明 |
|------|---|------|
| kernel | 'rbf' | 径向基函数（高斯核）|
| C | 10 | 正则化参数（平衡误差与模型复杂度）|
| gamma | 'scale' | 核宽度系数 |

### RBF核函数公式

$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$

其中：
- $x, x'$: 两个样本点
- $\gamma$: 控制影响范围的参数
- $||x - x'||$: 两点间欧氏距离

### 优缺点

**优点：**
- 理论基础扎实
- 适合高维数据
- 泛化性能好

**缺点：**
- 对大规模数据训练慢
- 需要特征缩放
- 参数调整较难

---

## 代码详解

### 训练随机森林

```python
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model
```

**逐步说明：**

1. **模型初始化**
   ```python
   RandomForestRegressor(n_estimators=100, ...)
   ```
   - 创建100棵决策树的随机森林

2. **模型训练**
   ```python
   model.fit(X_train, y_train)
   ```
   - `X_train`: 训练特征矩阵 (n_samples × 2048)
   - `y_train`: 训练目标值 (n_samples,)
   - 模型学习特征与目标值的关系

### 训练SVR

```python
def train_svr(X_train, y_train):
    model = SVR(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)
    return model
```

**参数说明：**

- `kernel='rbf'`: 使用径向基函数
- `C=10`: 控制对误差的容忍度
  - C越大：训练误差越小，可能过拟合
  - C越小：模型更简单，可能欠拟合
- `gamma='scale'`: 自动设置为1/(n_features × X.var())

### 模型保存与加载

```python
def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
```

**说明：**
- `pickle`: Python的序列化模块
- `'wb'`: 二进制写入模式
- `'rb'`: 二进制读取模式

---

## 模型选择建议

### 何时使用随机森林？

- 需要特征重要性
- 数据量中等
- 需要快速训练
- 特征可能有缺失

### 何时使用SVR？

- 样本量不是特别大
- 特征维度高
- 需要良好的泛化
- 特征已标准化

### QSAR中的常用选择

1. **探索性研究**: 随机森林（快速，提供特征重要性）
2. **小型数据集**: SVR（泛化好）
3. **大型数据集**: XGBoost/LightGBM（更高效）
4. **复杂模式**: 深度学习（需要大量数据）

---

## 模型评估指标

在后续的评估模块中会详细介绍，这里简要说明：

### R² (决定系数)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- 范围: (-∞, 1]
- 1 = 完美预测
- 0 = 用均值预测
- <0 = 比均值预测还差

### RMSE (均方根误差)

$$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

- 与目标值同单位
- 越小越好

---

## 延伸学习

- 统计学习理论
- 集成学习方法
- 超参数调优
- 交叉验证
