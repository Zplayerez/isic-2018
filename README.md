# ISIC-2018 皮肤病变分割项目代码说明文档

## 项目概述

本项目使用深度学习技术实现了对ISIC 2018皮肤病变图像的自动分割。主要基于ResNet50架构构建了全卷积网络(FCN)，通过迁移学习方法解决医学图像分割问题。项目涵盖了完整的机器学习流程，包括数据预处理、模型构建、训练、评估和预测。系统在病变边界分割任务(Task 1)上取得了较好效果。

## 项目结构

```
isic-2018/
├── .gitignore                   # Git 忽略文件配置
├── README.md                    # 项目说明文档
├── requirements.txt             # Python 依赖包列表
├── dataset.py                   # 数据集处理相关代码
├── dataset_converter.py         # 数据集格式转换脚本
├── eval.py                      # 模型评估脚本
├── image-analysis.py            # 图像分析相关脚本
├── img_data.csv                 # 图像数据的CSV描述文件
├── isic-resnet.ipynb            # ResNet模型相关的Jupyter Notebook
├── isic-task2.ipynb             # 任务2相关的Jupyter Notebook
├── isic-vgg.ipynb               # VGG模型相关的Jupyter Notebook
├── model_resnet.py              # ResNet模型结构定义
├── model_vgg.py                 # VGG模型结构定义
├── predict.py                   # 预测脚本
├── train.py                     # 训练脚本
├── visualization.py             # 可视化相关脚本
├── img/                         # 各类图片（模型结构、结果可视化等）
├── ISIC/                        # ISIC原始数据及标签
│   ├── train/                   # 训练图片
│   ├── train-resized/           # 调整尺寸后的训练图片
│   ├── labels/                  # 训练图片标签
│   └── task2-labels/            # 任务2的标签
├── output/                      # 输出结果文件夹
│   ├── external_test/           # 外部测试集结果
│   ├── test/                    # 测试集结果
│   ├── evaluation/              # 评估结果
│   ├── task1-resnet-04292032/   # 不同时间/参数下的实验结果
│   ├── task1-resnet-04292023/
│   └── task1-resnet-04292009/
└── test_data/                   # 测试数据相关
    ├── img_data.csv             # 测试集图片描述
    ├── predictions/             # 预测结果
    ├── images/                  # 原始测试图片
    ├── images-resized/          # 调整尺寸后的测试图片
    └── labels/                  # 测试图片标签
```

## 核心源码文件详解

### 1. dataset_converter.py

该文件用于转换原始ISIC数据集格式，确保文件名和格式标准化:

```python
# 关键功能
def main():
    # 创建输出目录
    if not os.path.exists(Out_Images_path):
        os.makedirs(Out_Images_path)
    if not os.path.exists(Out_Masks_path):
        os.makedirs(Out_Masks_path)
    
    # 获取图像文件列表并排序
    image_names = sorted(os.listdir(Origin_Images_path))
    mask_names = sorted(os.listdir(Origin_Masks_path))
    
    # 处理图像文件 - 转换为JPG
    for i, image_name in enumerate(image_names):
        image = Image.open(os.path.join(Origin_Images_path, image_name))
        image = image.convert('RGB')
        new_image_name = f"ISIC_{i}.jpg"
        image.save(os.path.join(Out_Images_path, new_image_name))
    
    # 处理掩码文件 - 保持PNG格式
    for i, mask_name in enumerate(mask_names):
        mask = Image.open(os.path.join(Origin_Masks_path, mask_name))
        new_mask_name = f"ISIC_{i}_segmentation.png"
        mask.save(os.path.join(Out_Masks_path, new_mask_name))
```

运行此脚本会将原始数据格式化并重命名为一致的格式，便于后续处理。

### 2. image-analysis.py

该文件包含两个主要类：`ScanTask`和`CountTask`，用于图像分析和处理。对于我们的复现流程，主要使用`ScanTask`：

```python
class ScanTask(Task):
    def __init__(self, args):
        super().__init__(args)
        self.task1_only = args.task1_only
        self.image_folder = args.image_folder
        self.label_folder = args.label_folder
        self.task2_label_folder = args.task2_label_folder

    def getFileList(self):
        # 获取所有.jpg图像的路径
        return glob(os.path.join(self.image_folder, 'ISIC_*.jpg'))

    def processFile(self, filename):
        # 处理单个文件
        # 1. 提取图像编号
        imgno = re.match('.*ISIC_(\\d*)\\.jpg', filename).group(1)
        
        # 2. 读取图像和对应的掩码
        img = cv2.imread(filename)
        label_path = os.path.join(self.label_folder, f'ISIC_{imgno}_segmentation.png')
        label = cv2.imread(label_path)
        
        # 3. 调整图像大小
        if self.args.resize:
            resized_img = cv2.resize(img, (self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.args.out, f'{imgno}.png'), resized_img)
            resized_label = cv2.resize(label, (self.args.resize, self.args.resize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(self.args.out, f'{imgno}_mask.png'), resized_label)
        
        # 4. 计算ROI(感兴趣区域)
        # 为任务2准备数据（如果未指定task1_only）
```

该脚本接受以下参数：

- `-resize 224`: 将图像调整为224×224像素
- `-out`: 指定输出目录
- `-task1_only`: 仅处理任务1相关数据
- `-filename`: 输出的元数据文件名

### 3. model_resnet.py

定义了基于ResNet50的FCN网络结构：

```python
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()
        # 使用预训练的ResNet50作为编码器
        self.num_classes = num_classes
        self.resnet = models.resnet50(pretrained=True)
        
        # 默认冻结所有ResNet参数
        for i, param in enumerate(self.resnet.parameters()):
            param.requires_grad = False
        
        # 解码器部分（转置卷积）
        self.a_convT2d = nn.ConvTranspose2d(2048, 256, 4, 2, 1)  # 7x7→14x14
        self.b_convT2d = nn.ConvTranspose2d(1280, 128, 4, 2, 1)  # 14x14→28x28
        self.c_convT2d = nn.ConvTranspose2d(640, 64, 4, 2, 1)    # 28x28→56x56
        self.convT2d3 = nn.ConvTranspose2d(320, num_classes, 4, 4, 0)  # 56x56→224x224
    
    def setTrainableLayers(self, trainable_layers):
        # 设置哪些层可训练
        for name, node in self.resnet.named_children():
            unlock = name in trainable_layers
            for param in node.parameters():
                param.requires_grad = unlock
    
    def forward(self, x):
        # 前向传播，包含编码、跳跃连接和解码
        skipConnections = {}
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        skipConnections[1] = x = self.resnet.layer1(x)  # [10, 256, 56, 56]
        skipConnections[2] = x = self.resnet.layer2(x)  # [10, 512, 28, 28]
        skipConnections[3] = x = self.resnet.layer3(x)  # [10, 1024, 14, 14]
        skipConnections[4] = x = self.resnet.layer4(x)  # [10, 2048, 7, 7]

        # 解码部分，通过转置卷积上采样
        x = self.a_convT2d(x)  # [10, 256, 14, 14]
        x = torch.cat((x, skipConnections[3]), 1)  # 跳跃连接
        
        x = self.b_convT2d(x)  # [10, 128, 28, 28]
        x = torch.cat((x, skipConnections[2]), 1)  # 跳跃连接
        
        x = self.c_convT2d(x)  # [10, 64, 56, 56]
        x = torch.cat((x, skipConnections[1]), 1)  # 跳跃连接
        
        x = self.convT2d3(x)  # [10, num_classes, 224, 224]
        x = nn.Sigmoid()(x)
        x = x.view(x.size()[0], -1, self.num_classes)
        
        return x
```

这个架构包含了标准的编码器-解码器结构，以及跳跃连接，用于保留不同尺度的空间信息。

### 4. dataset.py

管理数据加载和预处理:

```python
class LesionDataset(data.Dataset):
    # 输入图像标准化处理
    input_processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def __init__(self, x, y, imgnos, input_preprocessor, augment=False):
        self.imgnos = imgnos
        self.y = y  # 掩码路径列表
        self.x = x  # 图像路径列表
        self.input_preprocessor = input_preprocessor
        self.augment = augment
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # 加载图像和掩码
        item = [self.imread(self.x[idx])] + [self.imread(y) for y in self.y[idx]]
 
        # 数据增强（如果启用）
        if self.augment:
            item = RandomAffine()(item)
            
        # 预处理
        x = self.input_preprocessor(item[0])
        if len(item) > 2:
            y = np.dstack([self.labelcvt(tgt) for tgt in item[1:]]).squeeze()
        else:
            y = self.labelcvt(item[1])

        return x, y, self.imgnos[idx]
```

此外，该文件还包含`LesionData`类，用于创建训练和验证数据加载器：

```python
class LesionData(object):
    def __init__(self):
        # 加载图像元数据
        self.table = {}
        with open('./img_data.csv', 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                self.table[row[0]] = (
                    (int(row[1]), int(row[2]), int(row[3])),  # shape
                    (int(row[4]), int(row[5]), int(row[6]), int(row[7]))  # ROI
                )
    
    def getTask1TrainingDataLoaders(self, val_percent=20, batch_size=10, augment=False):
        # 创建训练和验证数据加载器
        imgnos = self.getImgNos()
        random.shuffle(imgnos)

        # 拆分验证集
        numval = val_percent*len(imgnos)//100
        val_imgnos = imgnos[0: numval]
        train_imgnos = imgnos[numval:]
        
        # 创建验证集
        val_x = [f'./ISIC/train-resized/{n}.png' for n in val_imgnos]
        val_y = [[f'./ISIC/train-resized/{n}_mask.png'] for n in val_imgnos]
        val_dataset = LesionDataset(val_x, val_y, val_imgnos, LesionDataset.input_processor, augment=False)

        # 创建训练集
        train_x = [f'./ISIC/train-resized/{n}.png' for n in train_imgnos]
        train_y = [[f'./ISIC/train-resized/{n}_mask.png'] for n in train_imgnos]
        train_dataset = LesionDataset(train_x, train_y, train_imgnos, LesionDataset.input_processor, augment=augment)

        return (
            data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True),
            data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        )
```

### 5. train.py

实现模型训练和验证循环:

```python
def fit(net, train_loader, val_loader, criterion, optimizer, lrscheduler, measures, epoch, loss_vis):
    # 训练阶段
    net.train(True)
    train_loss = 0.
    epoch_size = len(train_loader)
    losses = []
    
    for i, items in enumerate(train_loader, 0):
        # 获取批次数据
        inputs = items[0]
        labels = items[1]
        
        # 将数据移至GPU
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # 梯度清零、前向传播、损失计算、反向传播、参数更新
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        losses.append(loss.item())
        
        # 定期验证并可视化
        if i % 30 == 0:
            loss_vis.plot_loss(np.mean(losses), (epoch_size * epoch) + i, 'train_loss')
            losses.clear()
            
            # 切换到评估模式进行验证
            net.train(False)
            val_loss, measurements = validate(val_loader, net, criterion, measures, epoch)
            loss_vis.plot_loss(val_loss, (epoch_size * epoch) + i, 'val_loss')
            net.train(True)
            
            # 学习率调整
            if lrscheduler:
                lrscheduler.step(val_loss)
            
            # 可视化评估指标
            for k in measures.keys():
                measures[k][1].plot_loss(measurements[k], (epoch_size * epoch) + i, k)
    
    # 最终测量结果
    measurements['train_loss'] = train_loss / epoch_size
    measurements['val_loss'] = val_loss
    return measurements
```

以及验证函数：

```python
def validate(data_loader, net, criterion, measures, epoch):
    val_loss = 0.
    measurements = {k:0. for k in measures.keys()}
    
    for i, items in enumerate(data_loader, 0):
        inputs = items[0]
        labels = items[1]
        
        # 将数据移至GPU
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # 计算评估指标
        for (k, mobj) in measures.items():
            m = mobj[0]  # 指标函数
            measurements[k] += m(outputs, labels).item()
        
        val_loss += loss.item()
    
    # 平均所有批次的结果
    for k in measures.keys():
        measurements[k] = measurements[k] / len(data_loader)
        
    return val_loss / len(data_loader), measurements
```

### 6. isic-resnet.ipynb

这个Jupyter笔记本结合了上述所有组件，逐步执行模型训练过程：

1. **导入依赖项和模型定义**

   ```python
   import os, time, cv2, numpy as np, torch, torchvision
   import torch.optim as optim
   from model_resnet import Net
   ```

2. **实例化模型**

   ```python
   net = Net(num_classes=1).cuda()
   print(net)
   ```

3. **加载数据集**

   ```python
   import dataset
   train_loader, val_loader = dataset.create_loaders(val_percent=20, batch_size=10)
   ```

4. **定义损失函数和评估指标**

   ```python
   def weighted_fscore_loss(prew=1, recw=1):
       def fscore_loss(y_true, y_pred):
           presci = precision(y_true, y_pred)
           rec = recall(y_true, y_pred)
           return -(prew+recw)*(presci * rec)/(prew*presci + recw*rec)
       return fscore_loss
   ```

5. **设置优化器和可视化工具**

   ```python
   optimizer = optim.Adam(net.parameters(), lr=learning_rate)
   lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
   
   # 可视化设置
   iou_vis = Visualization('IOU / Dice Coeff.')
   fpfn_vis = Visualization('False Positive / False Negative')
   fscore_vis = Visualization('Precision, Recall, F-Score')
   loss_vis = Visualization('Mean loss')
   ```

6. **训练模型**

   ```python
   measurement_log = []
   fscores = [(1,1)] * 25  # 损失函数权重配置
   trainable = [['layer2', 'layer3', 'layer4']] * 25  # 设置可训练层
   
   # 训练循环
   for epoch in range(25):
       net.setTrainableLayers(trainable[epoch])
       measurements = train.fit(net, train_loader, val_loader, 
                               weighted_fscore_loss(*(fscores[epoch])), 
                               optimizer, lrscheduler, measures, epoch, loss_vis)
       measurement_log.append(measurements)
       print(f"Epoch: {epoch}: ", end='')
       for k,v in measurements.items():
           print(f" {k}:{v:.5f}", end=',')
       print()
   ```

7. **保存模型和指标**

   ```python
   timestamp = time.strftime('%m%d%H%M')
   outputfolder = f'./output/task1-resnet-{timestamp}'
   os.mkdir(outputfolder)
   
   # 保存模型权重
   net.save(os.path.join(outputfolder, 'lesions.pth'))
   
   # 保存训练日志和指标
   with open(os.path.join(outputfolder, 'measurements.txt'), 'w') as f:
       wr = csv.writer(f)
       wr.writerow(["Epoch"] + list(measurements.keys()))
       for epoch, measurement in enumerate(measurement_log):
           wr.writerow([epoch] + list(measurement.values()))
   ```

### 7. predict.py

使用训练好的模型进行预测：

```python
class Predictor:
    def __init__(self, model_path, test_dir, output_dir):
        # 初始化预测器
        self.test_dir = test_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载模型
        self.net = Net().cuda()
        self.net.load(model_path)
        self.net.eval()
        
        # 定义输入预处理
        self.input_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def process_file(self, image_path):
        # 处理单个图像
        base_name = os.path.basename(image_path)
        base_path = os.path.splitext(base_name)[0]
        
        # 构造对应的掩码文件路径
        mask_path = os.path.join(self.test_dir, f"{base_path}_mask.png")
        
        # 读取图像和掩码
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
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
```

该脚本的`_makeAnnotatedImage`方法创建四联图可视化，包含原图、真实掩码、预测掩码和组合视图。

## 模型参数详解

### 网络架构参数

1. **编码器(ResNet50)**
   - 预训练参数: ImageNet权重
   - 冻结/解冻层: 在`isic-resnet.ipynb`中配置为`[['layer2', 'layer3', 'layer4']] * 25`，意味着只有后三层可训练
2. **解码器(转置卷积)**
   - a_convT2d: [2048, 256, 4, 2, 1] - 将7x7特征图上采样至14x14
   - b_convT2d: [1280, 128, 4, 2, 1] - 将14x14特征图上采样至28x28
   - c_convT2d: [640, 64, 4, 2, 1] - 将28x28特征图上采样至56x56
   - convT2d3: [320, 1, 4, 4, 0] - 将56x56特征图上采样至224x224输出图
3. **跳跃连接**
   - 三个跳跃连接分别连接编码器的layer1/2/3和解码器的对应层

### 训练超参数

```python
# isic-resnet.ipynb中的参数
learning_rate = 1e-3           # 学习率
batch_size = 10                # 批次大小
epochs = 25                    # 训练轮数
val_percent = 20               # 验证集百分比
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # Adam优化器
```

### 损失函数

```python
# 加权F-Score损失函数
def weighted_fscore_loss(prew=1, recw=1):
    def fscore_loss(y_true, y_pred):
        presci = precision(y_true, y_pred)  # 精确率
        rec = recall(y_true, y_pred)        # 召回率
        return -(prew+recw)*(presci * rec)/(prew*presci + recw*rec)
    return fscore_loss
```

- 在训练中使用配置: `fscores = [(1,1)] * 25`，表示精确率和召回率权重相等

## 模型执行流程

### 1. 数据预处理

```bash
# 数据集格式转换
python dataset_converter.py

# 图像分析与预处理 (训练集)
python image-analysis.py scan -resize 224 -out ./ISIC/train-resized -task1_only -filename img_data.csv

# 图像分析与预处理 (测试集)
python image-analysis.py scan -resize 224 -out ./test_data/images-resized -task1_only -filename ./test_data/img_data.csv -image_folder ./test_data/images -label_folder ./test_data/labels
```

这些步骤完成了以下工作:

- 标准化图像命名
- 调整图像大小为224×224像素
- 生成描述图像尺寸和ROI的元数据

### 2. 模型训练

在Jupyter笔记本中执行`isic-resnet.ipynb`的所有单元格，主要流程为:

1. **初始化模型**：创建并配置ResNet50-FCN模型

2. **加载数据**：使用`dataset.py`中的函数加载训练和验证数据

3. 训练循环

   ：

   - 将模型设置为训练模式
   - 对每个批次执行前向传播和反向传播
   - 定期验证并记录指标
   - 使用Visdom可视化训练进度

4. **保存模型**：将训练好的模型保存为`lesions.pth`文件

### 3. 模型预测

```bash
# 使用训练好的模型在测试集上进行预测
python predict.py --model ./output/task1-resnet-04292032/lesions.pth --input test_data/images-resized --output test_data/predictions
```

该步骤:

1. 加载训练好的模型
2. 处理测试集中的每张图像
3. 生成分割掩码和可视化结果
4. 输出评估指标

通过这个执行流程，项目实现了从数据预处理、模型训练到预测分析的完整闭环，成功应用深度学习技术解决了医学图像分割问题。
