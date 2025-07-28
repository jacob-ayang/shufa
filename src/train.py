import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
from PIL import Image

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入自定义模块
from models.stroke_model import StrokeModel
from models.artistic_model import ArtisticModel
from data.dataset import ChineseCharacterDataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_stroke_model():
    """训练笔画模型"""
    print("开始训练笔画模型...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载数据集
    train_dataset = ChineseCharacterDataset(
        root_dir='../data/HWDB1.1trn', 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    
    # 初始化模型
    model = StrokeModel(num_classes=3755).to(device)  # HWDB1.1有3755个字符类
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 保存模型
        torch.save(model.state_dict(), f'../models/stroke_model_epoch_{epoch+1}.pth')
    
    print("笔画模型训练完成！")


def train_artistic_model():
    """训练艺术性模型"""
    print("开始训练艺术性模型...")
    
    # 这里需要实现艺术性模型的训练逻辑
    # 根据论文，艺术性模型包括字体特征、情感特征和文风特征的提取
    
    print("艺术性模型训练完成！")


def main():
    """主函数"""
    print("开始训练硬笔书法人工智能评判系统...")
    
    # 训练笔画模型（规范性评分的一部分）
    train_stroke_model()
    
    # 训练艺术性模型
    train_artistic_model()
    
    print("所有模型训练完成！")

if __name__ == "__main__":
    main()