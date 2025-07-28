import numpy as np

# 加载保存的数据
loaded_data = np.load('all_parsed_data.npy', allow_pickle=True)
loaded_labels = np.load('all_parsed_data_labels.npy', allow_pickle=True)

# 显示数据的基本信息
print(f"数据形状: {loaded_data.shape}")
print(f"数据类型: {loaded_data.dtype}")
print(f"标签数量: {len(loaded_labels)}")

# 显示数据范围
print(f"数据范围: [{loaded_data.min()}, {loaded_data.max()}]")

# 显示前5个样本的前10个元素
print("\n前5个样本的前10个元素:")
for i in range(min(5, len(loaded_data))):
    print(f"样本 {i+1}: {loaded_data[i][:10]}")
    print(f"标签 {i+1}: {loaded_labels[i]}")

# 显示一些统计信息
print(f"\n数据集统计信息:")
print(f"  总样本数: {len(loaded_data)}")
print(f"  特征维度: {loaded_data.shape[1]}")
print(f"  数据类型: {loaded_data.dtype}")