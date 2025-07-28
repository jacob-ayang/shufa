"""
硬笔书法评分系统主模块
集成预处理、分割、识别、规范性评分和艺术性评分模块
"""

import cv2
import numpy as np
from src.preprocessing.image_preprocessing import preprocess_image
from src.segmentation.character_segmentation import segment_characters
from src.recognition.character_recognition import recognize_characters
from src.normative_scoring.normative_scorer import NormativeScorer
from src.artistic_scoring.artistic_scorer import ArtisticScorer


class ScoringSystem:
    """
    硬笔书法评分系统
    """
    
    def __init__(self):
        """
        初始化评分系统
        """
        self.normative_scorer = NormativeScorer()
        self.artistic_scorer = ArtisticScorer()
    
    def score(self, image_path, template_path=None):
        """
        对硬笔书法作品进行评分
        
    Args:
        image_path (str): 输入图像路径
        template_path (str): 模板图像路径（可选）
        
    Returns:
        dict: 评分结果
    """
        # 1. 图像预处理
        print("正在进行图像预处理...")
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        
        processed_image = preprocess_image(original_image)
        
        # 2. 字符分割
        print("正在进行字符分割...")
        characters, positions = segment_characters(processed_image)
        
        # 3. 字符识别
        print("正在进行字符识别...")
        recognized_chars = recognize_characters(characters)
        
        # 4. 规范性评分
        print("正在进行规范性评分...")
        # 如果提供了模板图像，则进行规范性评分
        if template_path:
            template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_image is not None:
                template_processed = preprocess_image(template_image)
                template_chars, template_positions = segment_characters(template_processed)
                
                # 对每个字符进行规范性评分
                normative_scores = []
                for i, (char, template_char) in enumerate(zip(characters, template_chars)):
                    score = self.normative_scorer.score(char, template_char)
                    normative_scores.append(score)
                
                # 计算平均规范性评分
                avg_normative_score = {
                    'stroke_score': np.mean([s['stroke_score'] for s in normative_scores]),
                    'structure_score': np.mean([s['structure_score'] for s in normative_scores]),
                    'passage_score': np.mean([s['passage_score'] for s in normative_scores]),
                    'total_score': np.mean([s['total_score'] for s in normative_scores])
                }
            else:
                # 如果无法加载模板图像，则使用默认评分
                avg_normative_score = {
                    'stroke_score': 80.0,
                    'structure_score': 80.0,
                    'passage_score': 80.0,
                    'total_score': 80.0
                }
        else:
            # 如果没有提供模板图像，则使用默认评分
            avg_normative_score = {
                'stroke_score': 80.0,
                'structure_score': 80.0,
                'passage_score': 80.0,
                'total_score': 80.0
            }
        
        # 5. 艺术性评分
        print("正在进行艺术性评分...")
        artistic_scores = []
        for char in characters:
            score = self.artistic_scorer.score(char)
            artistic_scores.append(score)
        
        # 计算平均艺术性评分
        avg_artistic_score = {
            'stroke_artistry_score': np.mean([s['stroke_artistry_score'] for s in artistic_scores]),
            'layout_coherence_score': np.mean([s['layout_coherence_score'] for s in artistic_scores]),
            'total_score': np.mean([s['total_score'] for s in artistic_scores])
        }
        
        # 6. 综合评分
        print("正在计算综合评分...")
        overall_score = {
            'normative_score': avg_normative_score['total_score'],
            'artistic_score': avg_artistic_score['total_score'],
            'total_score': (avg_normative_score['total_score'] + avg_artistic_score['total_score']) / 2
        }
        
        return {
            'normative_scores': avg_normative_score,
            'artistic_scores': avg_artistic_score,
            'overall_score': overall_score,
            'characters': recognized_chars,
            'character_count': len(characters)
        }
    
    def generate_report(self, scoring_results):
        """
        生成评分报告
        
    Args:
        scoring_results (dict): 评分结果
        
    Returns:
        str: 评分报告
    """
        report = f"""
硬笔书法评分报告
================

识别字符: {''.join(scoring_results['characters'])}
字符数量: {scoring_results['character_count']}

规范性评分:
  笔画特征: {scoring_results['normative_scores']['stroke_score']:.2f}分
  结构特征: {scoring_results['normative_scores']['structure_score']:.2f}分
  章法特征: {scoring_results['normative_scores']['passage_score']:.2f}分
  规范性总分: {scoring_results['normative_scores']['total_score']:.2f}分

艺术性评分:
  笔画形态: {scoring_results['artistic_scores']['stroke_artistry_score']:.2f}分
  布局协调: {scoring_results['artistic_scores']['layout_coherence_score']:.2f}分
  艺术性总分: {scoring_results['artistic_scores']['total_score']:.2f}分

综合评分: {scoring_results['overall_score']['total_score']:.2f}分
        """
        
        return report.strip()


def main():
    """
    主函数 - 演示评分系统使用方法
    """
    # 创建评分系统实例
    scoring_system = ScoringSystem()
    
    # 示例：对书法作品进行评分
    # 注意：需要提供实际的图像路径
    image_path = "data/raw/sample.jpg"
    template_path = "data/raw/template.jpg"
    
    try:
        # 进行评分
        results = scoring_system.score(image_path, template_path)
        
        # 生成并打印报告
        report = scoring_system.generate_report(results)
        print(report)
        
    except Exception as e:
        print(f"评分过程中出现错误: {e}")


if __name__ == "__main__":
    main()