import numpy as np

def show_dataset_info():
    """
    显示数据集的基本信息
    """
    # 加载数据
    print("正在加载数据...")
    vectors = np.load('all_parsed_data.npy', allow_pickle=True)
    labels = np.load('all_parsed_data_labels.npy', allow_pickle=True)
    
    # 显示基本信息
    print("\n=== 数据集信息 ===")
    print(f"总样本数: {len(vectors)}")
    print(f"特征维度: {vectors.shape[1]}")
    print(f"数据类型: {vectors.dtype}")
    print(f"标签数量: {len(labels)}")
    
    # 显示数据范围
    print(f"\n数据值范围: [{vectors.min()}, {vectors.max()}]")
    
    # 显示内存使用情况
    vector_memory = vectors.nbytes / (1024**3)  # GB
    label_memory = labels.nbytes / (1024**3)    # GB
    total_memory = vector_memory + label_memory
    
    print(f"\n内存使用情况:")
    print(f"  向量数据: {vector_memory:.2f} GB")
    print(f"  标签数据: {label_memory:.2f} GB")
    print(f"  总计: {total_memory:.2f} GB")
    
    # 显示一些样本标签信息
    print(f"\n标签信息示例:")
    unique_labels = set()
    for i in range(min(1000, len(labels))):  # 只检查前1000个标签以节省时间
        unique_labels.add(labels[i])
    
    print(f"  前1000个样本中的唯一标签数: {len(unique_labels)}")
    
    # 显示前几个标签
    print(f"\n前10个样本的标签:")
    for i in range(min(10, len(labels))):
        print(f"  样本 {i+1}: {labels[i]}")

if __name__ == "__main__":
    show_dataset_info()