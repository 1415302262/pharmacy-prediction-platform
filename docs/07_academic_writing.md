# 学术写作与论文规范

## 摘要（Abstract）

### 写作要点

1. **简明扼要**：200-300字
2. **结构清晰**：背景→方法→结果→结论
3. **无缩写**：首次出现时拼写全称

### 示例结构

```
This project demonstrates the application of machine learning in drug discovery
through a Quantitative Structure–Activity Relationship (QSAR) analysis of EGFR
inhibitors. Using molecular fingerprints and traditional machine learning
algorithms, we developed predictive models to estimate compound activity
based on structural features. The Random Forest model achieved R² = X.XXX
with RMSE = X.XXX, demonstrating the potential of AI-assisted drug screening.
```

---

## 标题（Title）

### 优秀标题特点

1. **准确**：反映研究内容
2. **简洁**：一般不超过15个单词
3. **专业**：使用领域术语

### 示例

**好：**
- "QSAR Modeling of EGFR Inhibitors Using Machine Learning"
- "Machine Learning Approaches for Drug Activity Prediction"

**避免：**
- "My QSAR Project"（太随意）
- "A Study of Something"（太模糊）

---

## 引言（Introduction）

### 写作逻辑

1. **研究背景**
   - 为什么这个靶点重要？
   - 现有方法有什么局限？

2. **研究目的**
   - 要解决什么问题？
   - 研究目标是什么？

3. **研究意义**
   - 有什么创新点？
   - 有什么应用价值？

### 写作技巧

- **从大到小**：从广泛背景切入，聚焦到具体问题
- **引用文献**：引用关键研究
- **明确目标**：最后一句明确研究目标

### 示例段落

```
Quantitative Structure–Activity Relationship (QSAR) is a computational
approach that correlates molecular structure with biological activity.
In modern drug discovery, AI-Driven Drug Discovery (AIDD) combines QSAR
principles with machine learning to accelerate the identification of
promising drug candidates. Epidermal Growth Factor Receptor (EGFR) is a
validated cancer therapeutic target. The availability of extensive public
data on EGFR inhibitors makes it an ideal target for QSAR modeling.
```

---

## 方法（Methods）

### 写作要点

1. **详细但不过于详细**
2. **清晰描述步骤**
3. **关键参数要说明**

### 结构建议

```
1. Data Source
   - 数据来源
   - 数据格式

2. Feature Engineering
   - 特征提取方法
   - 关键参数

3. Model Development
   - 使用的算法
   - 超参数设置

4. Evaluation
   - 评估指标
   - 数据划分
```

### 参数表格示例

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Radius | 2 | Captures substructures within 2 bonds |
| nBits | 2048 | Sufficient dimensionality |
| n_estimators | 100 | Balance between performance and speed |

---

## 结果（Results）

### 写作要点

1. **客观陈述**：不要过度解释
2. **数据支持**：用具体数字
3. **图表引用**：清晰引用

### 图表说明

**散点图说明示例：**
```
Figure 1 shows the predicted vs. true pIC50 values for the Random Forest
model. The diagonal red line represents perfect prediction. The R² value of
0.XXX indicates that the model explains XX.X% of the variance in the data.
```

**特征重要性说明示例：**
```
Figure 3 displays the top 20 most important fingerprint bits. The most
significant feature corresponds to [specific substructure], which is
consistent with known pharmacophore patterns for EGFR inhibitors.
```

---

## 讨论（Discussion）

### 写作要点

1. **解释结果**：为什么是这个结果？
2. **对比研究**：与文献对比
3. **承认局限**：诚实说明不足
4. **未来方向**：后续研究

### 讨论框架

```
1. 结果解释
   - 模型性能好坏的原因
   - 特征重要性的意义

2. 与文献对比
   - 与其他研究的结果对比
   - 方法上的优劣

3. 局限性
   - 数据量限制
   - 方法局限

4. 未来工作
   - 数据扩展
   - 方法改进
   - 实验验证
```

---

## 结论（Conclusion）

### 写作要点

1. **总结主要发现**
2. **强调贡献**
3. **简洁有力**

### 示例

```
This project demonstrates a complete AI-driven drug discovery workflow:
data curation, feature engineering, model development, and evaluation. The
best model achieved [specific metrics], demonstrating predictive capability.
These results illustrate how machine learning can accelerate drug discovery,
reducing the time and cost of identifying promising therapeutic candidates.
```

---

## 文献引用（References）

### 引用格式（本项目使用）

**期刊论文：**
```
Gaulton, A., et al. (2017). The ChEMBL database in 2017.
Nucleic Acids Research, 45(D1), D945-D954.
```

**书籍：**
```
Vapnik, V. (1999). The Nature of Statistical Learning Theory.
Springer.
```

### 引用时机

- **方法**：引用方法来源
- **数据**：引用数据来源
- **对比**：引用对比的研究

---

## 图表规范

### 图标题（Figure Caption）

```
Figure 1. Random Forest predicted vs. true pIC50 values. The diagonal
red line represents perfect prediction (y = x). Points closer to the line
indicate better predictions. R² = 0.XXX, RMSE = X.XXX.
```

### 图标题要素

1. **简洁描述**：图是什么
2. **图例说明**：解释符号/颜色
3. **关键数值**：重要指标值

### 坐标轴标签

```
xlabel: True pIC50 (units)
ylabel: Predicted pIC50 (units)
```

### 图例

- **简洁**：用简短词语
- **清晰**：使用标准符号
- **位置**：不遮挡数据

---

## 数字与单位

### 数字格式

- **小数位数**：保持一致（如：0.7542, 0.6231）
- **千位分隔符**：使用逗号（如：10,000）
- **范围表示**：用连字符（如：5-10）

### 单位表示

- **与数值间有空格**：10 nM, 25 °C
- **上标/下标**：正确使用（如：IC50, pIC50）
- **斜体**：变量用斜体（如：R²）

---

## 缩写规范

### 首次出现

```
Quantitative Structure–Activity Relationship (QSAR)
```

### 后续使用

```
QSAR models showed good performance.
```

### 常用缩写

| 缩写 | 全称 |
|------|------|
| QSAR | Quantitative Structure–Activity Relationship |
| ML | Machine Learning |
| AI | Artificial Intelligence |
| EGFR | Epidermal Growth Factor Receptor |
| IC50 | Half maximal inhibitory concentration |
| AIDD | AI-Driven Drug Discovery |

---

## 表格规范

### 表题（Table Title）

```
Table 1. Model Performance Comparison
```

### 表结构

| Model | R² | RMSE |
|-------|-----|------|
| Random Forest | 0.XXX | X.XXX |
| SVR | 0.XXX | X.XXX |

### 表格要点

1. **对齐整齐**：数字对齐
2. **单位明确**：如果有的话
3. **有效数字**：一致的小数位数

---

## 语言风格

### 时态

| 部分 | 时态 |
|------|------|
| 引言 | 一般现在时 |
| 方法 | 一般过去时 |
| 结果 | 一般过去时 |
| 讨论 | 一般现在时 |

### 语气

- **客观**：避免主观评价
- **学术**：使用正式语言
- **清晰**：避免歧义

---

## 常见错误

1. **过度声称**：避免使用"证明"、"首次"等词
2. **数据不足**：结论要有数据支持
3. **逻辑跳跃**：论证要有逻辑链条
4. **语法错误**：仔细检查语法
5. **引用错误**：确保引用准确

---

## 写作工具

### 英文写作

- **Grammarly**：语法检查
- **Hemingway Editor**：简化表达
- **ChatGPT**：语言润色（注意学术诚信）

### 中文写作

- **Grammarly for Chinese**：语法检查
- **有道写作**：语法检查和润色

### 格式工具

- **LaTeX**：专业学术排版
- **Word**：通用文档编辑
- **Markdown**：轻量级格式

---

## 查重说明

学术写作需要避免抄袭：

1. **直接引用**：加引号并标注来源
2. **改写**：用自己的话表达
3. **引用来源**：引用关键观点

---

## 本项目报告要求

### 格式

- 2页学术报告
- Markdown格式
- 包含图表

### 必含部分

1. Abstract
2. Introduction
3. Dataset
4. Feature Engineering
5. Machine Learning Models
6. Model Evaluation
7. Results and Discussion
8. Conclusion
9. References
