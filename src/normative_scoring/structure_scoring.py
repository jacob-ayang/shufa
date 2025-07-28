"""
结构特征评分模块
实现论文中提到的结构相似度计算（宽高比、重心位置、笔画位置相似度）
"""

import cv2
import numpy as np


def calculate_aspect_ratio(contour):
    """
    计算宽高比
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        
    Returns:
        float: 宽高比
    """
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / (h + 1e-6)
    return aspect_ratio


def calculate_centroid(contour):
    """
    计算重心位置
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        
    Returns:
        tuple: 重心坐标 (cx, cy)
    """
    moments = cv2.moments(contour)
    if moments['m00'] == 0:
        return 0, 0
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def calculate_stroke_positions(contour):
    """
    计算笔画位置
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        
    Returns:
        list: 笔画位置坐标列表
    """
    # 简化实现：使用轮廓上的关键点作为笔画位置
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    positions = [(int(point[0][0]), int(point[0][1])) for point in approx]
    return positions


def calculate_structure_similarity(input_contour, template_contour):
    """
    计算结构相似度
    
    Args:
        input_contour (numpy.ndarray): 输入字符轮廓
        template_contour (numpy.ndarray): 模板字符轮廓
        
    Returns:
        dict: 各项相似度得分
    """
    # 1. 宽高比相似度
    input_aspect = calculate_aspect_ratio(input_contour)
    template_aspect = calculate_aspect_ratio(template_contour)
    aspect_ratio_similarity = 1 - abs(input_aspect - template_aspect) / max(input_aspect, template_aspect, 1e-6)
    
    # 2. 重心位置相似度
    input_centroid = calculate_centroid(input_contour)
    template_centroid = calculate_centroid(template_contour)
    centroid_distance = np.sqrt((input_centroid[0] - template_centroid[0])**2 + 
                               (input_centroid[1] - template_centroid[1])**2)
    # 假设最大可能距离为图像对角线长度
    max_distance = np.sqrt(64**2 + 64**2)  # 假设图像大小为64x64
    centroid_similarity = 1 - centroid_distance / max_distance
    
    # 3. 笔画位置相似度
    input_positions = calculate_stroke_positions(input_contour)
    template_positions = calculate_stroke_positions(template_contour)
    
    if len(input_positions) > 0 and len(template_positions) > 0:
        # 计算位置差异
        min_positions = min(len(input_positions), len(template_positions))
        position_distances = []
        
        for i in range(min_positions):
            dist = np.sqrt((input_positions[i][0] - template_positions[i][0])**2 + 
                          (input_positions[i][1] - template_positions[i][1])**2)
            position_distances.append(dist)
        
        avg_position_distance = np.mean(position_distances)
        position_similarity = 1 - avg_position_distance / max_distance
    else:
        position_similarity = 0.0
    
    # 加权融合
    weights = [0.3, 0.4, 0.3]  # 宽高比、重心位置、笔画位置的权重
    overall_similarity = (weights[0] * aspect_ratio_similarity + 
                         weights[1] * centroid_similarity + 
                         weights[2] * position_similarity)
    
    return {
        'aspect_ratio_similarity': aspect_ratio_similarity,
        'centroid_similarity': centroid_similarity,
        'position_similarity': position_similarity,
        'overall_similarity': overall_similarity
    }


def score_structure_features(input_contour, template_contour):
    """
    结构特征评分
    
    Args:
        input_contour (numpy.ndarray): 输入字符轮廓
        template_contour (numpy.ndarray): 模板字符轮廓
        
    Returns:
        float: 结构相似度得分 (0-100)
    """
    similarities = calculate_structure_similarity(input_contour, template_contour)
    score = similarities['overall_similarity'] * 100
    return score