import os
import zipfile
import shutil
from config import *


def extract_dataset(zip_path, extract_to):
    """
    解压数据集
    :param zip_path: ZIP文件路径
    :param extract_to: 解压目录
    """
    print(f"正在解压 {zip_path} 到 {extract_to}")
    
    # 创建解压目录
    os.makedirs(extract_to, exist_ok=True)
    
    # 解压文件
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"解压完成: {zip_path}")


def extract_all_datasets():
    """解压所有数据集"""
    print("开始解压所有数据集...")
    
    # 创建数据目录
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 定义数据集映射
    datasets = {
        'HWDB1.0trn.zip': 'HWDB1.0trn',
        'HWDB1.0tst.zip': 'HWDB1.0tst',
        'HWDB1.1trn.zip': 'HWDB1.1trn',
        'HWDB1.1tst.zip': 'HWDB1.1tst',
        'OLHWDB1.0trn.zip': 'OLHWDB1.0trn',
        'OLHWDB1.0tst.zip': 'OLHWDB1.0tst',
        'Pot1.2Test.zip': 'Pot1.2Test',
        'Pot1.2Train.zip': 'Pot1.2Train'
    }
    
    # 解压每个数据集
    for zip_file, folder_name in datasets.items():
        zip_path = os.path.join(RAW_DATA_DIR, zip_file)
        extract_to = os.path.join(DATA_DIR, folder_name)
        
        # 检查ZIP文件是否存在
        if os.path.exists(zip_path):
            # 如果解压目录已存在，先删除
            if os.path.exists(extract_to):
                shutil.rmtree(extract_to)
            
            # 解压
            extract_dataset(zip_path, extract_to)
        else:
            print(f"警告: 找不到ZIP文件 {zip_path}")
    
    print("所有数据集解压完成！")


def main():
    """主函数"""
    extract_all_datasets()


if __name__ == "__main__":
    main()