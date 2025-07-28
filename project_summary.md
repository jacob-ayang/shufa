# 硬笔书法评分系统项目摘要

## 项目概述

本项目实现了一个基于计算机视觉和深度学习技术的硬笔书法智能评分系统，能够对硬笔书法作品进行规范性和艺术性两个维度的自动评分。系统通过模拟人类专家的评分过程，从技术角度量化书法作品的质量。

## 技术架构

### 核心模块

1. **图像预处理模块** (`src/preprocessing`)
   - 灰度化处理
   - 自适应二值化
   - 中值滤波去噪
   - 骨架提取算法

2. **字符分割模块** (`src/segmentation`)
   - 基于骨架的单字分割算法
   - KMeans聚类分组
   - 图像归一化处理

3. **字符识别模块** (`src/recognition`)
   - 基于CIRM框架的单字识别
   - ResNet50特征提取网络
   - 相似度计算算法

4. **规范性评分模块** (`src/normative_scoring`)
   - 笔画特征评分（PBOD算法）
   - 结构特征评分（宽高比、重心位置、笔画位置）
   - 章法特征评分（文字大小一致性、间隔均匀性）

5. **艺术性评分模块** (`src/artistic_scoring`)
   - 笔画形态评分（粗细变化、流畅性）
   - 布局协调性评分（字间搭配、行间协调）

6. **评估模块** (`src/evaluation`)
   - 准确率计算
   - 相关性分析（皮尔逊、斯皮尔曼）
   - MAE/RMSE误差计算

7. **工具模块** (`src/utils`)
   - 数据加载和预处理
   - 可视化功能

### 主评分系统

`src/scoring_system.py`集成了所有核心模块，提供统一的评分接口。

## 算法实现

### 字符分割算法
基于论文中的骨架拆分和K-means聚类算法实现，能够有效分离粘连字符。

### 字符识别算法
采用CIRM（Character Identification using Radon and Moment Features）框架，结合ResNet50深度学习模型进行特征提取和识别。

### 规范性评分算法
1. **笔画特征**：使用PBOD（Position-Based Orientation Descriptor）算法计算笔画相似度
2. **结构特征**：综合考虑字符的宽高比、重心位置和笔画位置分布
3. **章法特征**：评估文字大小一致性和间隔均匀性

### 艺术性评分算法
1. **笔画形态**：分析笔画的粗细变化和流畅性
2. **布局协调**：评估字间搭配和行间协调性

## 项目特点

1. **完整的评分体系**：同时考虑规范性和艺术性两个维度
2. **模块化设计**：各功能模块独立，便于维护和扩展
3. **可视化支持**：提供处理过程和结果的可视化功能
4. **评估机制**：内置评估模块，支持性能分析

## 技术栈

- Python 3.8+
- OpenCV
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib
- Scikit-learn

## 使用方法

```python
from src.scoring_system import ScoringSystem

# 创建评分系统实例
scoring_system = ScoringSystem()

# 对书法作品进行评分
results = scoring_system.score("path/to/your/image.jpg", "path/to/template.jpg")

# 生成评分报告
report = scoring_system.generate_report(results)
print(report)
```

## 后续改进方向

1. **优化字符分割算法**：提高对复杂粘连字符的分割准确率
2. **增强字符识别模型**：使用更大规模的数据集训练更准确的识别模型
3. **完善评分算法**：根据更多书法专家的评分标准优化评分算法
4. **增加用户界面**：开发图形用户界面，提升用户体验
5. **扩展支持更多字体**：支持楷书、行书、草书等不同字体的评分