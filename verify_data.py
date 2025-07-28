import numpy as np

# 加载保存的数据
loaded_data = np.load('parsed_data.npy')

# 显示数据的基本信息
print(f"数据形状: {loaded_data.shape}")
print(f"数据类型: {loaded_data.dtype}")
print(f"数据范围: [{loaded_data.min()}, {loaded_data.max()}]")

# 显示前5个样本的前10个元素
print("\n前5个样本的前10个元素:")
for i in range(5):
    print(f"样本 {i+1}: {loaded_data[i][:10]}")