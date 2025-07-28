# 手写汉字识别数据集解析工具

这个项目提供了一个用于解析手写汉字识别数据集（HWDB1.0）中MPF格式文件的工具。

## 项目结构

- `parse_mpf.py`: 主要的解析器实现
- `parse_all_mpf.py`: 批量解析所有MPF文件的脚本
- `verify_data.py`: 验证保存数据的脚本
- `verify_all_data.py`: 验证合并数据的脚本
- `dataset_info.py`: 显示数据集信息的脚本
- `train_example.py`: 机器学习训练示例脚本
- `show_project_structure.py`: 显示项目文件结构的脚本
- `requirements.txt`: 项目依赖列表
- `data/`: 存放数据集的目录
- `parsed_data.npy`: 单个文件解析后的数据文件
- `all_parsed_data.npy`: 所有文件合并解析后的数据文件
- `all_parsed_data_labels.npy`: 所有文件的标签数据文件

## 安装依赖

可以使用以下命令安装项目所需的依赖项：

```
pip install -r requirements.txt
```

## 查看项目结构

可以使用以下命令查看项目的文件结构：

```
python show_project_structure.py
```

## 使用方法

### 解析单个MPF文件

1. 确保已安装Python和NumPy库
2. 将MPF数据文件放在`data/HWDB1.0trn/HWDB1.0trn/`目录下
3. 运行解析脚本：
   ```
   python parse_mpf.py
   ```
4. 验证解析结果：
   ```
   python verify_data.py
   ```

### 批量解析所有MPF文件

1. 确保已安装Python和NumPy库
2. 将所有MPF数据文件放在`data/HWDB1.0trn/HWDB1.0trn/`目录下
3. 运行批量解析脚本：
   ```
   python parse_all_mpf.py
   ```
4. 验证合并后的解析结果：
   ```
   python verify_all_data.py
   ```
5. 查看数据集信息：
   ```
   python dataset_info.py
   ```

### 运行训练示例

1. 确保已安装Python和NumPy库
2. 确保已经运行过批量解析脚本生成数据文件
3. 运行训练示例脚本：
   ```
   python train_example.py
   ```

注意：训练示例脚本使用了一个简化的K近邻算法实现，仅用于演示目的。实际的汉字识别任务需要更复杂的预处理和模型。

## 解析器功能

- 解析MPF文件头信息
- 解析样本数据（标签和特征向量）
- 将解析结果保存为NumPy格式文件

## 输出信息

解析器会输出以下信息：
- 文件头详细信息
- 前5个样本的详细信息
- 保存的NumPy文件信息