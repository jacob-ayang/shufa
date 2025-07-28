"""
章法特征评分模块
实现论文中提到的章法相似度计算（文字大小相对一致性、文字间隔均匀性）
""

import cv2
import numpy as np


def calculate_character_areas(characters):
    """
    计算字符的实际面积
    
    Args:
        characters (list): 字符图像列表
        
    Returns:
        list: 字符面积列表
    """
    areas = []
    for char in characters:
        # 字符面积可以通过像素点数量计算
        area = np.sum(char > 0)
        areas.append(area)
    return areas


def calculate_area_similarity(input_areas, template_areas):
    """
    计算文字大小相对一致性
    
    Args:
        input_areas (list): 输入字符面积列表
        template_areas (list): 模板字符面积列表
        
    Returns:
        float: 文字大小相对一致性系数
    """
    if len(input_areas) == 0 or len(template_areas) == 0:
        return 0.0
    
    # 计算面积比例
    min_len = min(len(input_areas), len(template_areas))
    ratios = []
    
    for i in range(min_len):
        if template_areas[i] > 0:
            ratio = input_areas[i] / template_areas[i]
            ratios.append(ratio)
    
    if len(ratios) == 0:
        return 0.0
    
    # 计算方差
    ratios = np.array(ratios)
    mean_ratio = np.mean(ratios)
    variance = np.var(ratios)
    
    # 计算一致性系数 (简化实现)
    consistency = 1 / (1 + variance)
    return consistency


def calculate_character_intervals(characters, positions):
    """
    计算字符之间的间隔
    
    Args:
        characters (list): 字符图像列表
        positions (list): 字符位置列表
        
    Returns:
        list: 字符间隔列表
    """
    if len(positions) < 2:
        return []
    
    intervals = []
    for i in range(len(positions) - 1):
        # 计算相邻字符在水平方向上的最小距离
        interval = abs(positions[i+1][0] - positions[i][0])
        intervals.append(interval)
    
    return intervals


def calculate_interval_uniformity(intervals):
    """
    计算文字间隔均匀性
    
    Args:
        intervals (list): 字符间隔列表
        
    Returns:
        float: 文字间隔均匀性系数
    """
    if len(intervals) == 0:
        return 0.0
    
    # 计算间隔的方差
    intervals = np.array(intervals)
    mean_interval = np.mean(intervals)
    if mean_interval == 0:
        return 0.0
    
    variance = np.var(intervals)
    
    # 计算均匀性系数 (简化实现)
    uniformity = 1 / (1 + variance / (mean_interval**2))
    return uniformity


def score_passage_features(input_characters, input_positions, 
                          template_characters, template_positions):
    """
    章法特征评分
    
    Args:
        input_characters (list): 输入字符图像列表
        input_positions (list): 输入字符位置列表
        template_characters (list): 模板字符图像列表
        template_positions (list): 模板字符位置列表
        
    Returns:
        dict: 章法特征评分结果
    """
    # 1. 计算字符面积
    input_areas = calculate_character_areas(input_characters)
    template_areas = calculate_character_areas(template_characters)
    
    # 2. 计算文字大小相对一致性
    size_consistency = calculate_area_similarity(input_areas, template_areas)
    
    # 3. 计算字符间隔
    input_intervals = calculate_character_intervals(input_characters, input_positions)
    template_intervals = calculate_character_intervals(template_characters, template_positions)
    
    # 4. 计算文字间隔均匀性
    input_uniformity = calculate_interval_uniformity(input_intervals)
    template_uniformity = calculate_interval_uniformity(template_intervals)
    
    # 5. 加权融合
    weights = [0.45, 0.55]  # 大小一致性、间隔均匀性的权重
    passage_score = weights[0] * size_consistency + weights[1] * input_uniformity
    
    return {
        'size_consistency': size_consistency,
        'interval_uniformity': input_uniformity,
        'passage_score': passage_score
    }


def normalize_passage_score(score, mean=0.5, std=0.1):
    """
    归一化章法特征评分到0-100分
    
    Args:
        score (float): 原始评分
        mean (float): 样本均值
        std (float): 样本标准差
        
    Returns:
        float: 归一化后的评分 (0-100)
    """
    normalized_score = (score - mean) / std * 10 + 50
    # 限制在0-100范围内
    normalized_score = max(0, min(100, normalized_score))
    return normalized_score