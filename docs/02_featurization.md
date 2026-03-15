# 特征工程学习文档

## 什么是特征工程？

特征工程是将原始数据转换为机器学习模型可以理解和处理的数值特征的过程。在QSAR（定量构效关系）中，核心任务是将分子的化学结构表示为数值向量。

## 核心概念

### SMILES (Simplified Molecular Input Line Entry System)

SMILES是一种用ASCII字符串表示分子结构的规范。

**示例：**
- 甲烷: `C`
- 乙烷: `CC`
- 乙醇: `CCO`
- 苯: `c1ccccc1`

**优势：**
- 便于计算机处理
- 紧凑的文本格式
- 唯一性（规范化后）

### 分子指纹 (Molecular Fingerprint)

分子指纹是分子结构的数值编码，表示分子中是否存在特定的子结构或化学特征。

## Morgan Fingerprint (ECFP)

### 原理

Morgan指纹（也称为扩展连接指纹，ECFP）是基于圆形的分子指纹。它通过以下步骤生成：

1. **原子初始化**：为每个原子分配初始标识符，基于原子类型、杂化状态、电荷等
2. **迭代更新**：在每一轮迭代中，每个原子的标识符结合其邻居原子的标识符进行更新
3. **哈希编码**：将更新后的标识符通过哈希函数映射到固定位数的比特向量

### 参数说明

| 参数 | 本项目设置 | 含义 |
|------|-----------|------|
| radius | 2 | 圆形半径，决定了子结构的大小 |
| nBits | 2048 | 比特向量的长度 |

**半径的含义：**
- radius = 0: 只考虑原子本身
- radius = 1: 考虑原子及其直接邻居
- radius = 2: 考虑原子及其两层的邻居（即包含最多6个原子的子结构）

**ECFP命名规则：**
- ECFP0: radius = 0
- ECFP2: radius = 1
- ECFP4: radius = 2 (本项目使用)
- ECFP6: radius = 3

## 代码详解

### 函数：smiles_to_fingerprint

```python
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
```

**功能：** 将SMILES字符串转换为Morgan指纹

**参数：**
- `smiles`: 分子的SMILES字符串
- `radius`: 指纹半径
- `n_bits`: 输出向量的位数

**返回：**
- `np.ndarray`: 长度为`n_bits`的二进制向量

**步骤解析：**

1. **SMILES转换为分子对象**
   ```python
   mol = Chem.MolFromSmiles(smiles)
   ```
   - `Chem.MolFromSmiles`是RDKit函数，将SMILES字符串解析为RDKit的分子对象
   - 如果SMILES无效，返回`None`

2. **错误处理**
   ```python
   if mol is None:
       return np.zeros(n_bits)
   ```
   - 如果SMILES无效，返回全零向量
   - 避免程序崩溃

3. **生成指纹**
   ```python
   fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
   ```
   - `GetMorganFingerprintAsBitVect`: 生成二进制Morgan指纹
   - 返回`ExplicitBitVect`对象

4. **转换为numpy数组**
   ```python
   return np.array(fp)
   ```
   - 转换为标准numpy数组，便于后续机器学习处理

### 函数：load_and_featurize

```python
def load_and_featurize(data_path: str) -> tuple:
```

**功能：** 加载数据集并批量转换为特征矩阵

**返回：**
- `X`: 特征矩阵，形状为 (n_samples, 2048)
- `y`: 目标值数组
- `df`: 原始数据框

**步骤解析：**

1. **读取CSV文件**
   ```python
   df = pd.read_csv(data_path)
   ```
   - 使用pandas读取CSV
   - 假设CSV包含`smiles`和`pIC50`列

2. **批量特征提取**
   ```python
   X = np.array([smiles_to_fingerprint(s) for s in df['smiles']])
   ```
   - 使用列表推导式批量处理
   - `np.array`将列表转换为numpy数组

3. **提取目标值**
   ```python
   y = df['pIC50'].values
   ```
   - 提取pIC50列作为目标值

## 为什么使用Morgan指纹？

### 优势

1. **可解释性**: 每个比特位对应特定的子结构
2. **固定维度**: 所有分子生成相同长度的向量
3. **计算效率**: 快速生成，适合大规模数据
4. **广泛应用**: 药物发现领域的标准方法

### 与其他方法对比

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| Morgan指纹 | 二进制，基于子结构 | 传统ML，速度快 |
| MACCS指纹 | 二进制，预定义键 | 快速筛选 |
| 物理化学描述符 | 数值，计算属性 | 需要物化属性 |
| 分子图 | 图结构 | 深度学习/GNN |

## 常见问题

**Q: 为什么选择2048位？**
A: 2048是一个平衡点。太短可能导致哈希冲突（不同子结构映射到同一位），太长会增加计算开销和稀疏性。2048是常用的标准设置。

**Q: 比特位越多越好吗？**
A: 不一定。过高的维度会导致：
- 计算时间增加
- 模型过拟合风险
- 特征稀疏性增加

**Q: 如何选择radius？**
A: 通常根据任务需求：
- radius = 1 (ECFP2): 捕获较小的结构模式
- radius = 2 (ECFP4): 捕获中等大小的结构（常用）
- radius = 3 (ECFP6): 捕获较大的结构模式

## 延伸学习

- RDKit文档: https://www.rdkit.org/
- 摩根指纹论文: Rogers & Hahn (2010), J. Chem. Inf. Model.
- 其他分子表示: 3D描述符、量子化学描述符、图神经网络
