#--------------------------------------------------------#
#   该文件用于调整ISIC2018数据集的图片格式和文件名
#--------------------------------------------------------#
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

#--------------------------------------------------------#
#   原始图像和掩码路径
#   输出图像和掩码路径
#--------------------------------------------------------#
Origin_Images_path = "/home/ubuntu/Workspace/isic2018/test/images"
Origin_Masks_path = "/home/ubuntu/Workspace/isic2018/test/masks"
Out_Images_path = "/home/ubuntu/Workspace/isic-2018/test_data/images"
Out_Masks_path = "/home/ubuntu/Workspace/isic-2018/test_data/labels"

if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(Out_Images_path):
        os.makedirs(Out_Images_path)
    if not os.path.exists(Out_Masks_path):
        os.makedirs(Out_Masks_path)
    
    # 获取图像文件列表并排序
    image_names = sorted(os.listdir(Origin_Images_path))
    mask_names = sorted(os.listdir(Origin_Masks_path))
    
    # 确保图像和掩码数量相同
    assert len(image_names) == len(mask_names), "图像和掩码数量不匹配！"
    
    print("正在处理图像文件...")
    for i, image_name in enumerate(tqdm(image_names)):
        # 处理图像文件 - 转换为JPG
        image = Image.open(os.path.join(Origin_Images_path, image_name))
        image = image.convert('RGB')  # 确保图像是RGB格式
        new_image_name = f"ISIC_{i}.jpg"
        image.save(os.path.join(Out_Images_path, new_image_name))
    
    print("正在处理掩码文件...")
    for i, mask_name in enumerate(tqdm(mask_names)):
        # 处理掩码文件 - 保持PNG格式
        mask = Image.open(os.path.join(Origin_Masks_path, mask_name))
        new_mask_name = f"ISIC_{i}_segmentation.png"
        mask.save(os.path.join(Out_Masks_path, new_mask_name))
    
    print("转换完成！")
    print(f"图像已保存到 {Out_Images_path}")
    print(f"掩码已保存到 {Out_Masks_path}")