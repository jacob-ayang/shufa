"""
笔画特征评分模块
实现论文中提到的笔画分割(PBOD算法)和笔画相似度计算
"""

import cv2
import numpy as np
from scipy.spatial.distance import euclidean


def detect_crossing_points(contour):
    """
    使用PBOD算法检测交叉点
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        
    Returns:
        list: 交叉点坐标列表
    """
    # 简化实现：检测轮廓中的角点作为交叉点
    corners = cv2.goodFeaturesToTrack(
        contour.astype(np.uint8), 
        maxCorners=20, 
        qualityLevel=0.01, 
        minDistance=10
    )
    
    if corners is not None:
        return [(int(point[0][0]), int(point[0][1])) for point in corners]
    return []


def merge_regions(contour, crossing_points):
    """
    区域合并
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        crossing_points (list): 交叉点坐标列表
        
    Returns:
        list: 合并后的区域列表
    """
    # 简化实现：根据交叉点将轮廓分割为多个区域
    regions = []
    
    if len(crossing_points) == 0:
        regions.append(contour)
        return regions
    
    # 根据交叉点分割轮廓
    for i in range(len(crossing_points)):
        start_point = crossing_points[i]
        end_point = crossing_points[(i + 1) % len(crossing_points)]
        
        # 找到轮廓中两点之间的部分
        start_idx = find_closest_point_index(contour, start_point)
        end_idx = find_closest_point_index(contour, end_point)
        
        if start_idx < end_idx:
            region = contour[start_idx:end_idx+1]
        else:
            region = np.concatenate([contour[start_idx:], contour[:end_idx+1]])
        
        if len(region) > 0:
            regions.append(region)
    
    return regions


def find_closest_point_index(contour, point):
    """
    找到轮廓中距离给定点最近的点的索引
    
    Args:
        contour (numpy.ndarray): 轮廓点数组
        point (tuple): 目标点坐标
        
    Returns:
        int: 最近点的索引
    """
    distances = [euclidean(p[0], point) for p in contour]
    return np.argmin(distances)


def remove_distortion_points(region):
    """
    畸变点删除
    
    Args:
        region (numpy.ndarray): 区域点数组
        
    Returns:
        numpy.ndarray: 删除畸变点后的区域
    """
    # 简化实现：使用Douglas-Peucker算法简化轮廓
    epsilon = 0.1 * cv2.arcLength(region, True)
    simplified = cv2.approxPolyDP(region, epsilon, True)
    return simplified


def stroke_segment_recombination(regions):
    """
    笔画段重组
    
    Args:
        regions (list): 区域列表
        
    Returns:
        list: 重组后的笔画段列表
    """
    # 简化实现：将相邻区域根据方向和宽度一致性进行连接
    strokes = []
    
    for region in regions:
        if len(region) > 0:
            # 计算区域的基本属性
            x, y, w, h = cv2.boundingRect(region)
            
            stroke = {
                'points': region,
                'bbox': (x, y, w, h),
                'length': cv2.arcLength(region, False),
                'area': cv2.contourArea(region)
            }
            
            strokes.append(stroke)
    
    return strokes


def calculate_stroke_similarity(stroke1, stroke2):
    """
    计算单笔画相似度（长度、斜率、曲率三维度加权融合）
    
    Args:
        stroke1 (dict): 第一个笔画
        stroke2 (dict): 第二个笔画
        
    Returns:
        float: 相似度得分 (0-1)
    """
    # 长度相似度
    length1 = stroke1['length']
    length2 = stroke2['length']
    length_similarity = 1 - abs(length1 - length2) / max(length1, length2, 1e-6)
    
    # 斜率相似度（简化实现：使用边界框的宽高比）
    x1, y1, w1, h1 = stroke1['bbox']
    x2, y2, w2, h2 = stroke2['bbox']
    slope1 = h1 / (w1 + 1e-6)
    slope2 = h2 / (w2 + 1e-6)
    slope_similarity = 1 - abs(slope1 - slope2) / max(slope1, slope2, 1e-6)
    
    # 曲率相似度（简化实现：使用面积与周长的比值）
    area1 = stroke1['area']
    area2 = stroke2['area']
    perimeter1 = stroke1['length']
    perimeter2 = stroke2['length']
    curvature1 = area1 / (perimeter1 ** 2 + 1e-6)
    curvature2 = area2 / (perimeter2 ** 2 + 1e-6)
    curvature_similarity = 1 - abs(curvature1 - curvature2) / max(curvature1, curvature2, 1e-6)
    
    # 加权融合
    weights = [0.4, 0.3, 0.3]  # 长度、斜率、曲率的权重
    similarity = (weights[0] * length_similarity + 
                 weights[1] * slope_similarity + 
                 weights[2] * curvature_similarity)
    
    return similarity


def score_stroke_features(input_strokes, template_strokes):
    """
    笔画特征评分
    
    Args:
        input_strokes (list): 输入字符的笔画列表
        template_strokes (list): 模板字符的笔画列表
        
    Returns:
        float: 笔画相似度得分 (0-100)
    """
    if len(input_strokes) == 0 or len(template_strokes) == 0:
        return 0.0
    
    # 计算每对笔画之间的相似度
    similarities = []
    for input_stroke in input_strokes:
        stroke_similarities = []
        for template_stroke in template_strokes:
            similarity = calculate_stroke_similarity(input_stroke, template_stroke)
            stroke_similarities.append(similarity)
        
        # 取最大相似度作为该输入笔画的匹配得分
        if stroke_similarities:
            similarities.append(max(stroke_similarities))
    
    # 计算平均相似度
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # 映射到0-100分
    score = avg_similarity * 100
    
    return score