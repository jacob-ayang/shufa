"""
艺术性评分模块 - 布局协调性
实现整体布局协调性的艺术性评分（字间搭配、行间协调性）
"""

import cv2
import numpy as np


def calculate_character_spacing(characters, positions):
    """
    计算字符间距
    
    Args:
        characters (list): 字符图像列表
        positions (list): 字符位置列表
        
    Returns:
        list: 字符间距列表
    """
    if len(positions) < 2:
        return []
    
    spacings = []
    for i in range(len(positions) - 1):
        # 计算相邻字符在水平方向上的间距
        spacing = abs(positions[i+1][0] - positions[i][0])
        spacings.append(spacing)
    
    return spacings


def calculate_spacing_consistency(spacings):
    """
    计算字间搭配一致性
    
    Args:
        spacings (list): 字符间距列表
        
    Returns:
        float: 字间搭配一致性评分
    """
    if len(spacings) == 0:
        return 0.0
    
    # 计算间距的标准差
    spacings = np.array(spacings)
    mean_spacing = np.mean(spacings)
    if mean_spacing == 0:
        return 0.0
    
    std_spacing = np.std(spacings)
    
    # 计算一致性评分 (简化实现)
    consistency = 1 / (1 + std_spacing / mean_spacing)
    normalized_score = consistency * 100
    
    return normalized_score


def calculate_line_coherence(lines):
    """
    计算行间协调性
    
    Args:
        lines (list): 行信息列表，每行包含字符和位置信息
        
    Returns:
        float: 行间协调性评分
    """
    if len(lines) < 2:
        return 100.0  # 只有一行时认为是完美的
    
    # 计算每行的基准线
    baselines = []
    for line in lines:\        if line['positions']:
            # 简化实现：使用字符位置的平均y坐标作为基准线
            avg_y = np.mean([pos[1] for pos in line['positions']])
            baselines.append(avg_y)
    
    if len(baselines) < 2:
        return 100.0
    
    # 计算基准线之间的差异
    baselines = np.array(baselines)
    baseline_diff = np.std(baselines)
    
    # 计算协调性评分 (简化实现)
    coherence = 1 / (1 + baseline_diff / np.mean(baselines))
    normalized_score = coherence * 100
    
    return normalized_score


def score_layout_coherence(characters, positions, lines=None):
    """
    布局协调性评分
    
    Args:
        characters (list): 字符图像列表
        positions (list): 字符位置列表
        lines (list): 行信息列表（可选）
        
    Returns:
        dict: 布局协调性评分结果
    """
    # 1. 计算字符间距
    spacings = calculate_character_spacing(characters, positions)
    
    # 2. 计算字间搭配一致性
    spacing_consistency = calculate_spacing_consistency(spacings)
    
    # 3. 计算行间协调性
    line_coherence = calculate_line_coherence(lines) if lines else 100.0
    
    # 4. 加权融合
    weights = [0.4, 0.6]  # 字间搭配、行间协调性的权重
    coherence_score = weights[0] * spacing_consistency + weights[1] * line_coherence
    
    return {
        'spacing_consistency': spacing_consistency,
        'line_coherence': line_coherence,
        'coherence_score': coherence_score
    }