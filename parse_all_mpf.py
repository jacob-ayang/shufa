import os
import numpy as np
from parse_mpf import MPFParser

def parse_all_mpf_files(data_dir, output_file):
    """
    解析目录中的所有MPF文件并将结果合并保存
    :param data_dir: 包含MPF文件的目录路径
    :param output_file: 输出文件名
    """
    # 获取所有MPF文件
    mpf_files = [f for f in os.listdir(data_dir) if f.endswith('.mpf')]
    mpf_files.sort()  # 按文件名排序
    
    print(f"找到 {len(mpf_files)} 个MPF文件")
    
    # 存储所有数据
    all_vectors = []
    all_labels = []
    
    # 解析每个文件
    for i, filename in enumerate(mpf_files):
        file_path = os.path.join(data_dir, filename)
        print(f"处理文件 {i+1}/{len(mpf_files)}: {filename}")
        
        try:
            # 创建解析器实例
            parser = MPFParser(file_path)
            
            # 解析文件
            parser.parse()
            
            # 获取样本数据
            samples = parser.get_samples()
            
            # 提取向量和标签
            for sample in samples:
                all_vectors.append(sample['vector'])
                all_labels.append(sample['label'])
                
        except Exception as e:
            print(f"解析文件 {filename} 时出错: {e}")
            continue
    
    # 转换为NumPy数组
    if all_vectors:
        vectors_array = np.array(all_vectors)
        print(f"\n总共解析了 {len(all_vectors)} 个样本")
        print(f"数据形状: {vectors_array.shape}")
        print(f"数据类型: {vectors_array.dtype}")
        
        # 保存到文件
        np.save(output_file, vectors_array)
        print(f"\n已将向量数据保存到 '{output_file}' 文件中")
        
        # 保存标签到单独的文件
        labels_file = output_file.replace('.npy', '_labels.npy')
        np.save(labels_file, np.array(all_labels, dtype=object))
        print(f"已将标签数据保存到 '{labels_file}' 文件中")
    else:
        print("没有成功解析任何数据")

def main():
    # 数据目录
    data_dir = 'data/HWDB1.0trn/HWDB1.0trn'
    
    # 输出文件名
    output_file = 'all_parsed_data.npy'
    
    # 解析所有MPF文件
    parse_all_mpf_files(data_dir, output_file)

if __name__ == "__main__":
    main()