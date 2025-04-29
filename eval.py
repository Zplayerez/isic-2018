import os
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.optim as optim

from model_resnet import Net
import dataset

class Evaluator(object):

    def __init__(self, model_filepath, outputDir):
        self.outputDir = outputDir
        self.net = Net().cuda()
        self.net.load(model_filepath)
        self.net.eval()
        self.lesionData = dataset.LesionData()
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

    def _makeAnnotatedImage(self, x, y, label):    
        y = y.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()
        label = label.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()

        imgs = []

        # convert image to HSV for annotations
        orig = x.cpu().detach().numpy()
        orig = (orig + 1) * 127
        orig = orig.astype(np.uint8)
        orig = np.dstack((orig[0,:,:], orig[1,:,:], orig[2,:,:]))
        
        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Original', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
        imgs.append(img)

        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)

        # apply prediction and label markings
        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[label > .75] = 100 # BLUE
        s[label > .75] = 250
        cv2.putText(img, 'Label', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[y > .75] = 50 # GREEN
        s[y > .75] = 250
        cv2.putText(img, 'Prediction', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
        
        img = np.copy(orig)
        h = img[:,:,0]
        s = img[:,:,1]
        h[y > .75] += 50 # GREEN
        s[y > .75] = 250
        h[label > .75] += 100 # BLUE
        s[label > .75] = 250
        cv2.putText(img, 'Combined', (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0,0), 2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))

        final = np.hstack(imgs)
        return final

    def _saveAnnotatedOutput(self, x, y, label, idx, fname):
        final = self._makeAnnotatedImage(x, y, label)
        # 使用原始图像ID而不是循环索引命名文件
        base_name = os.path.splitext(os.path.basename(fname))[0]
        cv2.imwrite(os.path.join(self.outputDir, f'{base_name}_annotated.png'), final)

    def _saveMask(self, y, fname):
        y = y.reshape(dataset.IMAGE_HT, dataset.IMAGE_WD).cpu().detach().numpy()
        mask = np.zeros_like(y, dtype=np.uint8)
        mask[y>0.75] = 255
        
        # 修复尺寸问题
        shape = self.lesionData.getShape(fname)
        
        # 确保size是一个二元组
        if len(shape) >= 2:
            # 如果shape至少有两个元素，使用前两个
            size = (int(shape[1]), int(shape[0]))
        else:
            # 如果shape不足两个元素，使用默认值或原图大小
            size = (dataset.IMAGE_WD, dataset.IMAGE_HT)
        
        # 确保两个维度都是整数
        mask = cv2.resize(mask, size, cv2.INTER_CUBIC)
        
        # 使用.png扩展名确保能够写入
        base_filename = os.path.basename(fname)
        if not (base_filename.endswith('.png') or base_filename.endswith('.jpg')):
            base_filename = base_filename + '.png'
        
        output_path = os.path.join(self.outputDir, base_filename)
        
        cv2.imwrite(output_path, mask)

    def run(self, annotate=True, save_mask=True):
        loader = self.lesionData.getTask1EvalDataLoader(batch_size = 10)
        print('size: %d' % len(loader))
        
        # 添加性能指标变量
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_samples = 0
        
        for images, labels, fnames in loader:
            images = images.cuda()
            output = self.net(images)
            
            batch_size = images.size(0)
            for idx in range(batch_size):
                x, y, label, fname = images[idx], output[idx], labels[idx], fnames[idx]
                
                # 计算性能指标
                pred = (y > 0.75).float().cpu().detach()
                gt = label.float()
                
                # 计算Dice系数
                intersection = torch.sum(pred * gt)
                dice = (2. * intersection) / (torch.sum(pred) + torch.sum(gt))
                
                # 计算IoU
                union = torch.sum(pred) + torch.sum(gt) - intersection
                iou = intersection / (union + 1e-7)
                
                # 计算精确率和召回率
                precision = intersection / (torch.sum(pred) + 1e-7)
                recall = intersection / (torch.sum(gt) + 1e-7)
                
                # 累加指标
                total_dice += dice.item()
                total_iou += iou.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_samples += 1
                
                # 保存可视化和掩码
                if annotate:
                    self._saveAnnotatedOutput(x, y, label, idx, fname)
                if save_mask:
                    self._saveMask(y, fname)
        
        # 计算平均指标
        avg_dice = total_dice / total_samples
        avg_iou = total_iou / total_samples
        avg_precision = total_precision / total_samples
        avg_recall = total_recall / total_samples
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-7)
        
        # 输出结果
        print("\n===== 评估结果 =====")
        print(f"平均Dice系数: {avg_dice:.4f}")
        print(f"平均IoU: {avg_iou:.4f}")
        print(f"平均精确率: {avg_precision:.4f}")
        print(f"平均召回率: {avg_recall:.4f}")
        print(f"F1分数: {f1_score:.4f}")
        
        # 将结果保存到文件
        with open(os.path.join(self.outputDir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("===== 评估结果 =====\n")
            f.write(f"平均Dice系数: {avg_dice:.4f}\n")
            f.write(f"平均IoU: {avg_iou:.4f}\n")
            f.write(f"平均精确率: {avg_precision:.4f}\n")
            f.write(f"平均召回率: {avg_recall:.4f}\n")
            f.write(f"F1分数: {f1_score:.4f}\n")

    def sample(self):
        loader = self.lesionData.getTask1EvalDataLoader(batch_size = 10)
        
        # 简单地获取第一批次数据
        for images, labels, fnames in loader:
            images = images.cuda()
            output = self.net(images)
            samples = []
            for idx in range(min(10, len(images))):
                x, y, label, fname = images[idx], output[idx], labels[idx], fnames[idx]
                samples.append(self._makeAnnotatedImage(x, y, label))
            return samples  # 返回样本并退出循环
        
        # 如果没有数据
        return []


if __name__ == "__main__":
    evaluator = Evaluator('./output/task1-resnet-04292032/lesions.pth', './output/test')
    evaluator.sample()
    evaluator.run(annotate=True, save_mask=True)