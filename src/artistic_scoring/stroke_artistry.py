"""
艺术性评分模块 - 笔画形态
实现笔画形态的艺术性评分（笔画粗细变化、笔画形态流畅性）
"""

import cv2
import numpy as np
from scipy import ndimage


def calculate_stroke_thickness_variation(character_image):
    """
    计算笔画粗细变化
    
    Args:
        character_image (numpy.ndarray): 字符图像
        
    Returns:
        float: 笔画粗细变化评分
    """
    # 使用距离变换计算笔画粗细
    distance_transform = ndimage.distance_transform_edt(character_image > 0)
    
    # 计算距离变换的标准差作为粗细变化指标
    thickness_variation = np.std(distance_transform)
    
    # 归一化到0-100范围（简化实现）
    normalized_score = min(100, thickness_variation * 10)
    
    return normalized_score


def calculate_stroke_fluency(character_image):
    """
    计算笔画形态流畅性
    
    Args:
        character_image (numpy.ndarray): 字符图像
        
    Returns:
        float: 笔画形态流畅性评分
    """
    # 查找字符轮廓
    contours, _ = cv2.findContours(character_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # 计算轮廓的弧长
    total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    
    # 计算轮廓面积
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    
    if total_area == 0:
        return 0.0
    
    # 计算流畅性指标（周长与面积的比值）
    fluency_ratio = total_perimeter / total_area
    
    # 归一化到0-100范围（简化实现）
    normalized_score = max(0, min(100, 100 - fluency_ratio * 5))
    
    return normalized_score


def score_stroke_artistry(character_image):
    """
    笔画艺术性评分
    
    Args:
        character_image (numpy.ndarray): 字符图像
        
    Returns:
        dict: 笔画艺术性评分结果
    """
    # 计算笔画粗细变化评分
    thickness_score = calculate_stroke_thickness_variation(character_image)
    
    # 计算笔画形态流畅性评分
    fluency_score = calculate_stroke_fluency(character_image)
    
    # 加权融合
    weights = [0.4, 0.6]  # 粗细变化、流畅性的权重
    artistry_score = weights[0] * thickness_score + weights[1] * fluency_score
    
    return {
        'thickness_variation': thickness_score,
        'stroke_fluency': fluency_score,
        'artistry_score': artistry_score
    }