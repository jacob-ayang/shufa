"""
单字分割模块
实现论文中提到的基于骨架的单字分割算法
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_stroke_segments(skeleton):
    """
    骨架拆分为笔画段
    
    Args:
        skeleton (numpy.ndarray): 输入的骨架图像
        
    Returns:
        list: 笔画段列表
    """
    # 查找轮廓
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stroke_segments = []
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 提取笔画段
        segment = skeleton[y:y+h, x:x+w]
        stroke_segments.append({
            'segment': segment,
            'bbox': (x, y, w, h),
            'centroid': (x + w//2, y + h//2)
        })
    
    return stroke_segments


def calculate_centroids(stroke_segments):
    """
    计算笔画段重心
    
    Args:
        stroke_segments (list): 笔画段列表
        
    Returns:
        numpy.ndarray: 重心坐标数组
    """
    centroids = np.array([seg['centroid'] for seg in stroke_segments])
    return centroids


def cluster_characters(stroke_segments, n_clusters=None):
    """
    使用KMeans聚类将笔画段分组为单字
    
    Args:
        stroke_segments (list): 笔画段列表
        n_clusters (int): 聚类数量，如果为None则自动确定
        
    Returns:
        list: 分组后的单字列表
    """
    if len(stroke_segments) == 0:
        return []
    
    # 计算重心
    centroids = calculate_centroids(stroke_segments)
    
    # 确定聚类数量
    if n_clusters is None:
        # 简单的启发式方法：根据笔画段数量估计字符数
        n_clusters = max(1, len(stroke_segments) // 3)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(centroids)
    
    # 根据聚类标签分组笔画段
    characters = []
    for i in range(n_clusters):
        char_segments = [seg for j, seg in enumerate(stroke_segments) if labels[j] == i]
        characters.append(char_segments)
    
    return characters


def normalize_character_images(characters, target_size=(64, 64)):
    """
    归一化单字图像
    
    Args:
        characters (list): 分组后的单字列表
        target_size (tuple): 目标尺寸
        
    Returns:
        list: 归一化后的单字图像列表
    """
    normalized_chars = []
    
    for char_segments in characters:
        # 合并同一字符的所有笔画段
        if len(char_segments) == 0:
            continue
            
        # 计算包含所有笔画段的最小边界框
        min_x = min(seg['bbox'][0] for seg in char_segments)
        min_y = min(seg['bbox'][1] for seg in char_segments)
        max_x = max(seg['bbox'][0] + seg['bbox'][2] for seg in char_segments)
        max_y = max(seg['bbox'][1] + seg['bbox'][3] for seg in char_segments)
        
        # 创建字符图像
        char_image = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        
        # 将笔画段绘制到字符图像上
        for seg in char_segments:
            x, y, w, h = seg['bbox']
            segment = seg['segment']
            char_image[y-min_y:y-min_y+h, x-min_x:x-min_x+w] = segment
        
        # 调整大小
        normalized_char = cv2.resize(char_image, target_size)
        normalized_chars.append(normalized_char)
    
    return normalized_chars


def segment_characters(skeleton, n_clusters=None, target_size=(64, 64)):
    """
    完整的单字分割流程
    
    Args:
        skeleton (numpy.ndarray): 输入的骨架图像
        n_clusters (int): 聚类数量
        target_size (tuple): 目标尺寸
        
    Returns:
        list: 归一化后的单字图像列表
    """
    # 1. 骨架拆分为笔画段
    stroke_segments = extract_stroke_segments(skeleton)
    
    # 2. 使用KMeans聚类将笔画段分组为单字
    characters = cluster_characters(stroke_segments, n_clusters)
    
    # 3. 归一化单字图像
    normalized_chars = normalize_character_images(characters, target_size)
    
    return normalized_chars