import os
import sys
import argparse
import cv2
import numpy as np
import torch
import re
from tqdm import tqdm
from glob import glob
import torchvision.transforms as transforms

# 导入模型定义
sys.path.append('.')  # 确保能找到项目中的其他模块
from model_resnet import Net

# 图像大小常量
IMAGE_HT = 224
IMAGE_WD = 224

class Predictor:
    def __init__(self, model_path, test_dir, output_dir):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            test_dir: 测试图像和掩码所在目录
            output_dir: 输出目录
        """
        self.test_dir = test_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.net = Net().cuda()
        self.net.load(model_path)
        self.net.eval()
        
        # 定义输入预处理
        self.input_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _makeAnnotatedImage(self, x, y, label):
        """
        创建带注释的四联图
        
        Args:
            x: 输入图像
            y: 预测掩码
            label: 真实掩码
            
        Returns:
            合并后的四联图
        """
        y = y.reshape(IMAGE_HT, IMAGE_WD).cpu().detach().numpy()
        label = label.reshape(IMAGE_HT, IMAGE_WD)

        imgs = []

        # 转换图像为HSV用于注释
        orig = x.cpu().detach().numpy()
        orig = (orig + 1) * 127
        orig = orig.astype(np.uint8)
        orig = np.dstack((orig[0,:,:], orig[1,:,:], orig[2,:,:]))
        
        img = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        cv2.putText(img, 'Original', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        imgs.append(img)

        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)

        # 标记真实掩码
        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[label > .75] = 100  # 蓝色
        s[label > .75] = 250
        cv2.putText(img, 'Label', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

        # 标记预测掩码
        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[y > .75] = 50  # 绿色
        s[y > .75] = 250
        cv2.putText(img, 'Prediction', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        
        # 合并显示
        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[y > .75] += 50  # 绿色
        s[y > .75] = 250
        h[label > .75] += 100  # 蓝色
        s[label > .75] = 250
        cv2.putText(img, 'Combined', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

        # 水平拼接四张图像
        final = np.hstack(imgs)
        return final
    
    def _saveMask(self, y, output_path):
        """
        保存预测掩码
        
        Args:
            y: 预测掩码
            output_path: 输出路径
        """
        y = y.reshape(IMAGE_HT, IMAGE_WD).cpu().detach().numpy()
        mask = np.zeros_like(y, dtype=np.uint8)
        mask[y > 0.5] = 255
        cv2.imwrite(output_path, mask)
    
    def process_file(self, image_path):
        """
        处理单个图像文件
        
        Args:
            image_path: 图像文件路径
        """
        try:
            # 提取图像名称和基本路径
            base_name = os.path.basename(image_path)
            base_path = os.path.splitext(base_name)[0]
            
            # 如果文件名已包含"_mask"，说明这是掩码文件，跳过
            if "_mask" in base_path:
                return
            
            # 构造对应的掩码文件路径
            mask_path = os.path.join(self.test_dir, f"{base_path}_mask.png")
            
            # 检查掩码文件是否存在
            if not os.path.exists(mask_path):
                print(f"警告: 找不到对应的掩码文件 {mask_path}")
                return
            
            # 读取图像和掩码
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"警告: 无法读取图像 {image_path}")
                return
                
            if mask is None:
                print(f"警告: 无法读取掩码 {mask_path}")
                return
            
            # 预处理图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = self.input_processor(image_rgb)
            x = x.unsqueeze(0)  # 添加批次维度
            
            # 进行预测
            with torch.no_grad():
                x = x.cuda()
                y = self.net(x)
            
            # 生成并保存注释图
            label = torch.from_numpy(mask.astype(np.float32) / 255.0)
            annotated = self._makeAnnotatedImage(x[0], y[0], label)
            
            # 保存结果
            output_path = os.path.join(self.output_dir, f"{base_path}_annotated.png")
            cv2.imwrite(output_path, annotated)
            
            # 保存预测掩码
            pred_mask_path = os.path.join(self.output_dir, f"{base_path}_mask.png")
            self._saveMask(y[0], pred_mask_path)
            
        except Exception as e:
            print(f"处理文件 {image_path} 时出错: {str(e)}")
    
    def run(self):
        """运行预测"""
        # 获取所有图像文件
        image_files = glob(os.path.join(self.test_dir, "*.png")) + glob(os.path.join(self.test_dir, "*.jpg"))
        
        # 过滤掉掩码文件
        image_files = [f for f in image_files if "_mask" not in os.path.basename(f)]
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for image_file in tqdm(image_files, desc="处理图像"):
            self.process_file(image_file)
            
        print(f"处理完成! 结果保存在 {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测并生成带注释的图像')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--input', type=str, required=True, help='测试图像和掩码目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    predictor = Predictor(args.model, args.input, args.output)
    predictor.run()

if __name__ == "__main__":
    main()