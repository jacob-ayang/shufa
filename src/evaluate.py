import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入自定义模块
from models.stroke_model import StrokeModel
from models.artistic_model import ArtisticModel
from data.dataset import ChineseCharacterDataset
from config import *


def evaluate_stroke_model():
    """评估笔画模型"""
    print("开始评估笔画模型...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载测试数据集
    test_dataset = ChineseCharacterDataset(
        root_dir=HWDB1_1_TEST, 
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # 初始化模型
    model = StrokeModel(num_classes=NUM_CLASSES).to(device)
    
    # 加载训练好的模型权重
    model_path = os.path.join(MODELS_DIR, 'stroke_model_epoch_10.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print(f"未找到模型权重文件: {model_path}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 评估
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'笔画模型测试准确率: {accuracy:.2f}%')


def evaluate_artistic_model():
    """评估艺术性模型"""
    print("开始评估艺术性模型...")
    
    # 这里需要实现艺术性模型的评估逻辑
    
    print("艺术性模型评估完成！")


def main():
    """主函数"""
    print("开始评估硬笔书法人工智能评判系统...")
    
    # 评估笔画模型（规范性评分的一部分）
    evaluate_stroke_model()
    
    # 评估艺术性模型
    evaluate_artistic_model()
    
    print("所有模型评估完成！")

if __name__ == "__main__":
    main()