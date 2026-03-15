# ChEMBL数据库说明

## ChEMBL是什么？

ChEMBL是一个由欧洲生物信息学研究所（EBI）维护的、经过人工整理的、具有药物样性质的生物活性分子数据库。

**网址：** https://www.ebi.ac.uk/chembl/

---

## 数据来源

### 数据收集

1. **科学文献**：从同行评审的期刊中提取
2. **专利文档**：包含大量化合物的专利数据
3. **公开数据库**：整合其他公开的药物数据库

### 数据特点

- **人工整理**：数据经过专业人员审查，质量高
- **标准化**：使用统一的格式和标准
- **完整性**：包含结构、活性、靶点等信息

---

## 数据结构

### 主要数据表

| 表名 | 说明 |
|------|------|
| molecule_dictionary | 化合物基本信息 |
| compound_structures | 化合物结构（SMILES）|
| activities | 生物活性数据 |
| assays | 实验方法信息 |
| target_dictionary | 靶点（蛋白质）信息 |

### 本项目使用的数据字段

#### SMILES

**SMILES示例：**
```
Cc1ccc(cc1Nc2ncnc(Nc3ccc(Oc4cccc(c4)C)cc3)n2)NC(=O)c5ccc(cc5)C
```

**解释：**
- `C`: 碳原子
- `c`: 芳香碳
- `N`: 氮原子
- `O`: 氧原子
- `(`, `)`: 分支
- `1`, `2`: 环闭合数字
- `=`: 双键
- `#`: 三键

**SMILES规则：**
1. 氢原子默认省略（除非是离子或特殊状态）
2. 芳香原子用小写字母
3. 环通过数字闭合

#### pIC50

**定义：**
$$pIC50 = -\log_{10}(IC50)$$

其中IC50是导致50%抑制时的浓度（摩尔浓度）。

**换算关系：**

| pIC50 | IC50 | 活性评价 |
|-------|------|----------|
| 5 | 10 μM | 弱活性 |
| 6 | 1 μM | 中等活性 |
| 7 | 100 nM | 较强活性 |
| 8 | 10 nM | 强活性 |
| 9 | 1 nM | 极强活性 |

**为什么使用pIC50？**
1. **近似正态分布**：IC50呈对数正态分布，pIC50更接近正态
2. **数值范围适中**：pIC50在5-10之间，便于处理
3. **线性关系**：与结合能成线性关系

---

## 如何从ChEMBL获取数据

### 方法1：使用ChEMBL Web界面

1. 访问 https://www.ebi.ac.uk/chembl/
2. 在搜索框输入靶点名称（如"EGFR"）
3. 筛选数据类型（如"IK50"）
4. 导出数据

### 方法2：使用ChEMBL API（高级）

```python
import requests

# 查询EGFR抑制剂
target = "CHEMBL203"
url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?target_chembl_id={target}"

response = requests.get(url)
data = response.json()
```

### 方法3：使用chembl_webresource_client库

```python
from chembl_webresource_client.new_client import new_client

# 创建客户端
molecule = new_client.molecule
activity = new_client.activity
target = new_client.target

# 查询EGFR靶点
target_id = 'CHEMBL203'
activities = activity.filter(target_chembl_id=target_id,
                            assay_type="B",
                            pchembl_value__isnull=False)

# 转换为DataFrame
import pandas as pd
df = pd.DataFrame(list(activities))
```

---

## 本项目的模拟数据

`data/sample_data.csv`包含70个EGFR抑制剂模拟数据。

### 数据特征

- **化合物数量**: 70个
- **靶点**: EGFR（表皮生长因子受体）
- **活性范围**: pIC50 5.7 - 9.1
- **结构特点**: 含嘧啶核心、苯胺取代基等

### 分子结构模式

#### 核心：嘧啶环
```
   N
  / \
 C   N
 |   |
 C   N
  \ /
   N
```

#### 常见取代基
- 苯环
- 氯原子（Cl）
- 氟原子（F）
- 氧原子桥（O）
- 甲基（CH₃）

---

## 数据质量控制

### 数据清理步骤

在实际项目中，需要：

1. **去除无效SMILES**
   ```python
   from rdkit import Chem

   def is_valid_smiles(smiles):
       mol = Chem.MolFromSmiles(smiles)
       return mol is not None
   ```

2. **标准化SMILES**
   ```python
   from rdkit.Chem import MolStandardize

   def standardize_smiles(smiles):
       mol = Chem.MolFromSmiles(smiles)
       smi = MolStandardize.normalize.MolStandardize().norm(mol)
       return Chem.MolToSmiles(smi)
   ```

3. **去除重复**
   ```python
   df = df.drop_duplicates(subset=['smiles'])
   ```

4. **过滤异常值**
   ```python
   # 去除极端活性值
   df = df[(df['pIC50'] > 4) & (df['pIC50'] < 11)]
   ```

### 数据集划分

在真实项目中，建议：

- **训练集**: 60-70%
- **验证集**: 10-20%
- **测试集**: 20%

或使用**分层划分**（保持活性分布一致）：

```python
from sklearn.model_selection import train_test_split

# 将活性分为几个区间
bins = [0, 6, 7, 8, 10]
labels = [0, 1, 2, 3]
y_bins = pd.cut(y, bins=bins, labels=labels)

# 分层划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_bins, random_state=42
)
```

---

## 其他可用数据源

除了ChEMBL，还有：

| 数据库 | 特点 |
|--------|------|
| PubChem | 大型数据库，包含多种生物活性数据 |
| BindingDB | 专注于蛋白-配体结合数据 |
| DrugBank | FDA批准药物和实验性药物 |
| ChEBI | 化学实体生物本体论 |

---

## 数据集大小建议

对于QSAR项目：

| 数据集大小 | 建议方法 |
|-----------|----------|
| < 100 | 仅用于演示，不实际使用 |
| 100-500 | 传统ML（RF, SVR）|
| 500-5000 | 传统ML + 交叉验证 |
| 5000-50000 | 传统ML + 深度学习 |
| > 50000 | 深度学习 / 大规模模型 |

---

## 延伸阅读

- ChEMBL论文: Gaulton et al. (2017) NAR
- SMILES规范: Weininger (1988) J. Chem. Inf. Comput. Sci.
- 数据质量评估方法
