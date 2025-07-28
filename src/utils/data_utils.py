"""
工具模块 - 数据处理
实现数据加载、保存和预处理功能
"""

import os
import numpy as np
import cv2


def load_character_image(image_path, target_size=(64, 64)):
    """
    加载字符图像并预处理
    
    Args:
        image_path (str): 图像路径
        target_size (tuple): 目标尺寸
        
    Returns:
        numpy.ndarray: 处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    # 调整尺寸
    image = cv2.resize(image, target_size)
    
    # 二值化
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image


def load_dataset(data_dir, target_size=(64, 64)):
    """
    加载数据集
    
    Args:
        data_dir (str): 数据集目录
        target_size (tuple): 目标尺寸
        
    Returns:
        tuple: (images, labels) 图像和标签列表
    """
    images = []
    labels = []
    
    # 遍历数据目录
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 获取图像路径
                image_path = os.path.join(root, file)
                
                try:
                    # 加载图像
                    image = load_character_image(image_path, target_size)
                    images.append(image)
                    
                    # 从目录名获取标签
                    label = os.path.basename(root)
                    labels.append(label)
                except Exception as e:
                    print(f"加载图像时出错 {image_path}: {e}")
    
    return np.array(images), np.array(labels)


def save_processed_data(data, labels, data_path, labels_path):
    """
    保存处理后的数据
    
    Args:
        data (numpy.ndarray): 数据
        labels (numpy.ndarray): 标签
        data_path (str): 数据保存路径
        labels_path (str): 标签保存路径
    """
    np.save(data_path, data)
    np.save(labels_path, labels)
    print(f"数据已保存到 {data_path}")
    print(f"标签已保存到 {labels_path}")


def load_processed_data(data_path, labels_path):
    """
    加载处理后的数据
    
    Args:
        data_path (str): 数据路径
        labels_path (str): 标签路径
        
    Returns:
        tuple: (data, labels) 数据和标签
    """
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    return data, labels