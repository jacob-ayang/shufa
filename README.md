# 硬笔书法人工智能评判系统

这个项目实现了一个基于神经网络的硬笔书法人工智能评判系统，旨在解决硬笔书法评价中艺术性量化难题。

## 项目结构

- `data/` - 解压后的数据集
- `raw_data/` - 原始zip数据集文件
- `models/` - 训练好的模型
- `notebooks/` - Jupyter notebooks用于探索性数据分析和模型开发
- `src/` - 源代码
  - `src/data/` - 数据处理代码
  - `src/models/` - 模型定义和训练代码
  - `src/utils/` - 工具函数

## 数据集

本项目使用了以下数据集：

- HWDB1.0trn, HWDB1.0tst
- HWDB1.1trn, HWDB1.1tst
- OLHWDB1.0trn, OLHWDB1.0tst
- Pot1.2Test, Pot1.2Train

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集预处理

数据集已自动解压到`data/`目录中。如果需要重新解压，可以使用以下命令：

```bash
python src/utils/extract_data.py
```

## 探索性数据分析

可以使用Jupyter notebook进行数据集的探索性数据分析：

```bash
jupyter notebook notebooks/eda.ipynb
```

## 训练模型

```bash
python src/train.py
```

## 评估模型

```bash
python src/evaluate.py
```

## 项目架构

本项目基于论文中提出的双通道评价架构：

1. **规范性评价通道**：使用传统算法对笔画、结构、章法进行量化评分
2. **艺术性评价通道**：使用跨模态框架（CNN提取字体特征，LSTM提取书写动态特征，BERT提取文本风格特征）

### 规范性评分模型
- 笔画特征提取：使用StrokeModel进行字符识别
- 结构特征评分：基于字符的结构特征进行评分
- 章法特征评分：考虑整篇作品的布局一致性

### 艺术性评分模型
- 字体特征提取：使用CNN提取视觉特征
- 情感特征提取：使用LSTM提取书写动态特征
- 文风特征提取：使用BERT提取文本风格特征
- 跨模态风格对齐：将不同模态的特征进行对齐
- 综合艺术性评分：结合多种特征进行艺术性评分

## 模型特点

1. **动态权重调整**：根据作品的艺术水平动态调整规范性评分和艺术性评分的权重
2. **多模态融合**：结合视觉、动态和文本信息进行综合评价
3. **端到端训练**：支持端到端的模型训练和优化