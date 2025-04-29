import os, sys
from glob import glob
import csv
import numpy as np
import re
import cv2
import torch
import matplotlib.pyplot as plt
import threading
import concurrent.futures as futures
import itertools
import argparse
import multiprocessing as mp

class Task():
    def __init__(self, args):
        self.args = args
    
    def start(self):
        filelist = self.getFileList()

        # start processes with shared queue
        self.fileQueue = mp.Queue()
        procs = [mp.Process(target=Task.run, args=[self, x]) for x in range(self.args.num_procs)]
        for proc in procs:
            proc.start()

        # write items to queue
        for f in filelist:
            self.fileQueue.put(f)

        # write quit signal
        for proc in procs:
            self.fileQueue.put(None)

        # wait for threads to die
        for proc in procs:
            proc.join()

        # merge outputs
        if self.args.filename:
            with open(args.filename, 'w') as fout:
                for id in range(self.args.num_procs):
                    fname = re.sub('\\.csv', '_%d.csv'%id, self.args.filename)
                    with open(fname, 'r') as fin:
                        fout.write(fin.read())
                    os.unlink(fname)



    def run(self, id):
        if self.args.resize is not None:
            if not os.path.exists(self.args.out):
                os.mkdir(self.args.out)

        wr = None
        if self.args.filename is not None:
            fname = re.sub('\\.csv', '_%d.csv'%id, self.args.filename)
            f = open(fname, 'w')
            wr = csv.writer(f)

        while True:
            filename = self.fileQueue.get()
            if filename is None:
                break;

            rowdata = self.processFile(filename)

            if wr is not None:
                wr.writerow(rowdata)
        
        if self.args.filename is not None:
            f.close()

    def processFile(self, filename):
        return []

    def getFileList(self):
        return []

# Scan: list image dimensions, resize training and label image for task1. crop ROI, resize to 224x224 for task 2
class ScanTask(Task):
    def __init__(self, args):
        super().__init__(args)
        # 添加一个任务选项
        self.task1_only = args.task1_only
        self.image_folder = args.image_folder
        self.label_folder = args.label_folder
        self.task2_label_folder = args.task2_label_folder

    def getFileList(self):
        return glob(os.path.join(self.image_folder, 'ISIC_*.jpg'))

    def processFile(self, filename):
        rowdata = []

        try:
            img = cv2.imread(filename)
            if img is None:
                print(f"警告: 无法读取图像 {filename}")
                return []
                
            imgno = re.match('.*ISIC_(\\d*)\\.jpg', filename).group(1)
            label_path = os.path.join(self.label_folder, 'ISIC_{}_segmentation.png'.format(imgno))
            
            if not os.path.exists(label_path):
                print(f"警告: 找不到标签文件 {label_path}")
                return []
                
            label = cv2.imread(label_path)
            if label is None:
                print(f"警告: 无法读取标签 {label_path}")
                return []

            rowdata.append(imgno)
            rowdata.extend(img.shape)

            if self.args.resize is not None:
                resized_img = cv2.resize(img, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(self.args.out, '{}.png'.format(imgno)), resized_img)
                resized_label = cv2.resize(label, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(self.args.out, '{}_mask.png'.format(imgno)), resized_label)

            # 计算ROI
            top = 0
            left = 0
            bottom = label.shape[0]
            right = label.shape[1]

            label_binary = label[:,:,0] == 255
            mrows = np.argwhere(np.any(label_binary, axis=1))
            mcols = np.argwhere(np.any(label_binary, axis=0))
            if len(mrows) > 0 and len(mcols) > 0:
                top = np.min(mrows)
                left = np.min(mcols)
                bottom = np.max(mrows)
                right = np.max(mcols)

            rowdata.extend([left, top, right, bottom])

            # 如果不是仅处理任务1，并且需要处理任务2的数据
            if self.args.resize is not None and not self.task1_only:
                # 写入裁剪、缩放的图像
                roi_img = img[top:bottom, left:right]
                resized_img = cv2.resize(roi_img, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(self.args.out, 'roi_{}.png'.format(imgno)), resized_img)

                # 图像对应的属性掩码
                attr_mask_files = [
                    os.path.join(self.task2_label_folder, 'ISIC_{}_attribute_globules.png'.format(imgno)),
                    os.path.join(self.task2_label_folder, 'ISIC_{}_attribute_milia_like_cyst.png'.format(imgno)),
                    os.path.join(self.task2_label_folder, 'ISIC_{}_attribute_negative_network.png'.format(imgno)),
                    os.path.join(self.task2_label_folder, 'ISIC_{}_attribute_pigment_network.png'.format(imgno)),
                    os.path.join(self.task2_label_folder, 'ISIC_{}_attribute_streaks.png'.format(imgno))
                ]

                for j, attr_mask_file in enumerate(attr_mask_files):
                    # 安全地读取文件 - 如果文件不存在，则创建一个空白掩码
                    if os.path.exists(attr_mask_file):
                        attr_mask = cv2.imread(attr_mask_file)
                        if attr_mask is not None:
                            roi_mask = attr_mask[top:bottom, left:right]
                            resized_mask = cv2.resize(roi_mask, dsize=(self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(os.path.join(self.args.out, 'roi_{}_mask_{}.png'.format(imgno, j)), resized_mask)
                        else:
                            # 创建空白掩码
                            empty_mask = np.zeros((self.args.resize, self.args.resize, 3), dtype=np.uint8)
                            cv2.imwrite(os.path.join(self.args.out, 'roi_{}_mask_{}.png'.format(imgno, j)), empty_mask)
                    else:
                        # 创建空白掩码
                        empty_mask = np.zeros((self.args.resize, self.args.resize, 3), dtype=np.uint8)
                        cv2.imwrite(os.path.join(self.args.out, 'roi_{}_mask_{}.png'.format(imgno, j)), empty_mask)

            return rowdata
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            return []

# Count: count pixels in mask images
class CountTask(Task):
    def __init__(self, args):
        super().__init__(args)
        self.mask_folder = args.mask_folder

    def getFileList(self):
        return glob(os.path.join(self.mask_folder, '*mask*.png'))

    def processFile(self, filename):
        try:
            m = re.search('.*?(\d+)_mask(_(\d+))?', filename)
            if m:
                img = cv2.imread(filename)
                if img is None:
                    return []
                    
                total = np.size(img)
                ones = np.count_nonzero(img)
                imgno = m.groups()[0]
                classno = m.groups()[2] if m.groups()[2] is not None else -1
                return [imgno, classno, total, ones]
            return []
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset analysis tool')
    parser.add_argument('function', action='store', help='Scan: list image dimensions, resize training and label image for task1. crop ROI, resize to 224x224 for task 2\nCount: count pixels in mask images')
    parser.add_argument('-filename', action='store', default=None, help='filename of image comparison data')
    parser.add_argument('-resize', action='store', nargs='?', type=int, const=224, help='resize to specified size')
    parser.add_argument('-out', action='store', default='./ISIC/train-resized', help='location to output scaled images')
    parser.add_argument('-num_procs', action='store', type=int, default=8, help='number of child processes')
    parser.add_argument('-task1_only', action='store_true', help='只处理任务1，跳过任务2的属性掩码处理')
    # 添加新的参数
    parser.add_argument('-image_folder', action='store', default='./ISIC/train', help='图像文件所在文件夹')
    parser.add_argument('-label_folder', action='store', default='./ISIC/labels', help='标签文件所在文件夹')
    parser.add_argument('-task2_label_folder', action='store', default='./ISIC/task2-labels', help='任务2标签文件所在文件夹')
    parser.add_argument('-mask_folder', action='store', default='./ISIC/train-resized', help='掩码文件所在文件夹')

    args = parser.parse_args()

    if args.function == 'scan':
        ScanTask(args).start()
    
    elif args.function == 'count':
        CountTask(args).start()