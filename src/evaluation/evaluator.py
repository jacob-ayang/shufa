"""
评估模块
实现评分系统的评估功能，包括准确率、相关性等指标计算
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def calculate_accuracy(predicted_scores, actual_scores, threshold=5):
    """
    计算评分准确率
    
    Args:
        predicted_scores (list): 预测评分列表
        actual_scores (list): 实际评分列表
        threshold (float): 误差阈值
        
    Returns:
        float: 准确率
    """
    if len(predicted_scores) != len(actual_scores):
        raise ValueError("预测评分和实际评分长度不匹配")
    
    # 计算误差
    errors = np.abs(np.array(predicted_scores) - np.array(actual_scores))
    
    # 计算准确率
    correct_predictions = np.sum(errors <= threshold)
    accuracy = correct_predictions / len(predicted_scores)
    
    return accuracy


def calculate_correlation(predicted_scores, actual_scores):
    """
    计算评分相关性
    
    Args:
        predicted_scores (list): 预测评分列表
        actual_scores (list): 实际评分列表
        
    Returns:
        dict: 相关性结果
    """
    if len(predicted_scores) != len(actual_scores):
        raise ValueError("预测评分和实际评分长度不匹配")
    
    # 计算皮尔逊相关系数
    pearson_corr, pearson_p = pearsonr(predicted_scores, actual_scores)
    
    # 计算斯皮尔曼相关系数
    spearman_corr, spearman_p = spearmanr(predicted_scores, actual_scores)
    
    return {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    }


def calculate_mae(predicted_scores, actual_scores):
    """
    计算平均绝对误差(MAE)
    
    Args:
        predicted_scores (list): 预测评分列表
        actual_scores (list): 实际评分列表
        
    Returns:
        float: 平均绝对误差
    """
    if len(predicted_scores) != len(actual_scores):
        raise ValueError("预测评分和实际评分长度不匹配")
    
    # 计算MAE
    mae = np.mean(np.abs(np.array(predicted_scores) - np.array(actual_scores)))
    
    return mae


def calculate_rmse(predicted_scores, actual_scores):
    """
    计算均方根误差(RMSE)
    
    Args:
        predicted_scores (list): 预测评分列表
        actual_scores (list): 实际评分列表
        
    Returns:
        float: 均方根误差
    """
    if len(predicted_scores) != len(actual_scores):
        raise ValueError("预测评分和实际评分长度不匹配")
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((np.array(predicted_scores) - np.array(actual_scores)) ** 2))
    
    return rmse


class Evaluator:
    """
    评估器类
    """
    
    def __init__(self):
        """
        初始化评估器
        """
        pass
    
    def evaluate(self, predicted_scores, actual_scores):
        """
        评估评分系统性能
        
    Args:
        predicted_scores (list): 预测评分列表
        actual_scores (list): 实际评分列表
        
    Returns:
        dict: 评估结果
    """
        # 计算准确率 (误差在5分以内)
        accuracy = calculate_accuracy(predicted_scores, actual_scores, threshold=5)
        
        # 计算相关性
        correlation = calculate_correlation(predicted_scores, actual_scores)
        
        # 计算MAE
        mae = calculate_mae(predicted_scores, actual_scores)
        
        # 计算RMSE
        rmse = calculate_rmse(predicted_scores, actual_scores)
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def generate_report(self, evaluation_results):
        """
        生成评估报告
        
    Args:
        evaluation_results (dict): 评估结果
        
    Returns:
        str: 评估报告
    """
        report = f"""
评分系统评估报告
==================

准确率 (误差阈值=5分): {evaluation_results['accuracy']:.4f}
平均绝对误差 (MAE): {evaluation_results['mae']:.4f}
均方根误差 (RMSE): {evaluation_results['rmse']:.4f}

相关性分析:
  皮尔逊相关系数: {evaluation_results['correlation']['pearson_correlation']:.4f} (p={evaluation_results['correlation']['pearson_p_value']:.4f})
  斯皮尔曼相关系数: {evaluation_results['correlation']['spearman_correlation']:.4f} (p={evaluation_results['correlation']['spearman_p_value']:.4f})
        """
        
        return report.strip()