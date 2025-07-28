# 硬笔书法评分系统

基于计算机视觉和深度学习技术的硬笔书法智能评分系统，实现对硬笔书法作品的规范性和艺术性自动评分。

## 项目结构

```
shufa/
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
│   └── models/           # 模型数据
├── src/                  # 源代码目录
│   ├── preprocessing/    # 图像预处理模块
│   ├── segmentation/     # 字符分割模块
│   ├── recognition/      # 字符识别模块
│   ├── normative_scoring/# 规范性评分模块
│   ├── artistic_scoring/ # 艺术性评分模块
│   ├── evaluation/       # 评估模块
│   ├── utils/            # 工具模块
│   └── scoring_system.py # 主评分系统
├── notebooks/            # Jupyter笔记本
├── tests/                # 测试代码
├── models/               # 训练好的模型
├── requirements.txt      # 项目依赖
├── train_example.py      # 训练示例
└── README.md             # 项目说明
```

## 功能特性

1. **图像预处理**
   - 灰度化
   - 二值化
   - 去噪
   - 骨架提取

2. **字符分割**
   - 基于骨架的单字分割算法
   - KMeans聚类分组
   - 图像归一化

3. **字符识别**
   - 基于CIRM框架的单字识别
   - ResNet50特征提取

4. **规范性评分**
   - 笔画特征评分（PBOD算法）
   - 结构特征评分（宽高比、重心位置等）
   - 章法特征评分（文字大小一致性、间隔均匀性）

5. **艺术性评分**
   - 笔画形态评分（粗细变化、流畅性）
   - 布局协调性评分（字间搭配、行间协调）

6. **评估模块**
   - 准确率计算
   - 相关性分析
   - MAE/RMSE计算

## 安装依赖

```bash
pip install -r requirements.txt
```

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

## 训练模型

```bash
python train_example.py
```

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

## 论文参考

本系统基于以下论文实现：

1. 基于骨架拆分和K-means聚类的单字分割算法
2. 基于CIRM框架的单字识别算法
3. 基于PBOD算法的笔画特征评分方法
4. 综合考虑笔画、结构和章法特征的规范性评分模型
5. 融合笔画形态和布局协调性的艺术性评分方法