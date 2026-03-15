# 模型评估与可视化学习文档

## 评估流程

模型评估是验证模型性能的关键步骤，包括：
1. 在测试集上进行预测
2. 计算评估指标
3. 可视化预测结果
4. 分析特征重要性

---

## 评估指标

### R² (决定系数，Coefficient of Determination)

**公式：**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**含义：**
- 衡量模型解释了多少目标值的方差
- 分子是模型预测误差的平方和
- 分母是用均值预测的误差平方和

**数值解读：**

| R²值 | 解释 |
|------|------|
| 1.0 | 完美预测（现实中不可能）|
| 0.8-0.9 | 非常好的预测 |
| 0.6-0.8 | 良好的预测 |
| 0.4-0.6 | 一般的预测 |
| <0.4 | 较差的预测 |
| <0 | 比瞎猜还差 |

### RMSE (均方根误差，Root Mean Square Error)

**公式：**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**含义：**
- 预测误差的平均值（平方后取平均，再开方）
- 与目标值单位相同
- 对大误差更敏感（平方放大）

**pIC50中的RMSE解读：**
- RMSE = 0.5: 预测偏差约±0.5 pIC50单位
- RMSE = 1.0: 预测偏差约±1.0 pIC50单位（一个数量级）

---

## 预测 vs 真实值散点图

### 图形目的

散点图直观展示模型预测能力：
- 点越接近对角线，预测越准确
- 可观察预测偏差的模式（系统性高估/低估）
- 识别离群点

### 代码解析

```python
def plot_predictions(y_true, y_pred, r2, rmse, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
```

**设置画布：**
- `figsize=(6, 5)`: 图像尺寸（6英寸宽，5英寸高）
- `fig, ax`: 获取图形对象和坐标轴对象

```python
ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
```

**绘制散点：**
- `alpha=0.6`: 透明度（0=完全透明，1=不透明）
- `s=50`: 点的大小
- `edgecolors='k'`: 点的边缘颜色（黑色）
- `linewidth=0.5`: 边缘线宽

```python
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
```

**绘制对角线：**
- `x = y_true.min()`, `x = y_true.max()`: 从左下到右上
- `'r--'`: 红色虚线
- `lw=2`: 线宽

```python
ax.set_xlabel('True pIC50', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize=11, fontweight='bold')
```

**设置标签：**
- `fontsize=11`: 字体大小
- `fontweight='bold'`: 加粗

```python
ax.set_title(f'Prediction Performance\nR² = {r2:.3f}, RMSE = {rmse:.3f}', ...)
```

**设置标题：**
- `\n`: 换行
- `:.3f`: 保留3位小数

```python
ax.grid(True, alpha=0.3)
```

**显示网格：**
- `True`: 启用网格
- `alpha=0.3`: 网格线透明度

```python
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
```

**保存图像：**
- `tight_layout()`: 自动调整布局，防止标签重叠
- `dpi=300`: 分辨率（300 = 高质量）
- `bbox_inches='tight'`: 紧凑边界
- `close()`: 关闭图形，释放内存

---

## 特征重要性图（随机森林）

### 目的

特征重要性图展示哪些分子特征对预测贡献最大：
- 可解释性强（知道模型关注哪些子结构）
- 指导药物设计（优化重要特征对应的结构）
- 模型验证（重要特征应与药理学知识一致）

### 代码解析

```python
def plot_feature_importance(model, save_path, top_n=20):
```

**参数说明：**
- `model`: 训练好的模型
- `save_path`: 保存路径
- `top_n=20`: 显示前20个最重要的特征

```python
if not hasattr(model, 'feature_importances_'):
    return
```

**检查模型是否有特征重要性：**
- 只有随机森林等树模型有此属性
- SVR等模型没有，直接返回

```python
importance = model.feature_importances_
```

**获取特征重要性：**
- 返回所有2048个特征的重要性分数
- 分数范围约在0到1之间，总和为1

```python
indices = np.argsort(importance)[::-1][:top_n]
```

**排序：**
- `np.argsort`: 返回排序后的索引
- `[::-1]`: 反转（降序）
- `[:top_n]`: 取前top_n个

```python
fig, ax = plt.subplots(figsize=(8, 5))
```

**创建图形：** 宽8英寸，高5英寸（比散点图更宽，容纳标签）

```python
ax.bar(range(top_n), importance[indices], color='steelblue', edgecolor='black')
```

**绘制柱状图：**
- `range(top_n)`: x轴位置（0,1,2,...,19）
- `importance[indices]`: y轴高度
- `color='steelblue'`: 柱子颜色
- `edgecolor='black'`: 柱子边缘颜色

```python
ax.set_xticklabels(indices, rotation=45, ha='right')
```

**设置x轴标签：**
- `rotation=45`: 旋转45度
- `ha='right'`: 文字右对齐

---

## 评估函数详解

```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'r2': r2, 'rmse': rmse, 'y_pred': y_pred}
```

**逐步说明：**

1. **预测**
   ```python
   y_pred = model.predict(X_test)
   ```
   - 输入测试集特征
   - 输出预测的目标值

2. **计算R²**
   ```python
   r2 = r2_score(y_test, y_pred)
   ```
   - 使用scikit-learn内置函数

3. **计算RMSE**
   ```python
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   ```
   - 先算MSE（均方误差）
   - 再开方得RMSE

4. **返回结果**
   ```python
   return {'r2': r2, 'rmse': rmse, 'y_pred': y_pred}
   ```
   - 字典形式返回
   - 包含两个指标和预测值（用于绘图）

---

## 可视化风格说明

本项目采用"科研风格"的可视化：

### 设计原则

1. **简洁清晰**
   - 避免过多装饰元素
   - 标签明确
   - 配色协调

2. **高分辨率**
   - `dpi=300` 确保打印质量
   - 适合论文发表

3. **可读性**
   - 合适的字体大小
   - 加粗标题和坐标轴标签
   - 网格线辅助读数

4. **信息完整**
   - 图例完整
   - 评估指标显示在图中
   - 坐标轴单位明确

### 配色方案

| 用途 | 颜色 | 说明 |
|------|------|------|
| 散点 | 蓝色系 | 标准数据点 |
| 对角线 | 红色虚线 | 理想预测线 |
| 柱状图 | 钢蓝色 | 专业、中性 |
| 网格 | 浅灰色 | 辅助但不干扰 |

---

## 常见问题

**Q: R²是0.7，这算好吗？**
A: 在QSAR任务中，R²在0.6-0.8通常被认为是可接受的。这取决于数据集大小、数据质量和目标复杂性。

**Q: 为什么散点图不都在对角线上？**
A: 完美预测在现实中几乎不存在。误差来源于：
- 数据噪声（实验误差）
- 模型局限（无法捕捉所有关系）
- 特征不足（缺少重要描述符）

**Q: 特征重要性有什么用？**
A: 可以：
- 了解模型关注哪些结构
- 指导新分子的设计
- 验证模型的合理性（重要特征应与已知药理知识一致）

---

## 延伸学习

- 交叉验证方法
- 学习曲线分析
- 置信区间计算
- 残差分析
