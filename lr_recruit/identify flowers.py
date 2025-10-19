import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from PIL import Image,UnidentifiedImageError
from pathlib import Path
#数据清洗脚本，检查异常或损坏数据
class DataCleaner:
    def __init__(self,data_root_path,delete_enabled=False):
        self.DATA_ROOT=data_root_path
        self.DELETE_FILES=delete_enabled
        self.log_file=self.DATA_ROOT/"corrupted_files.log"
        self.total_checked=0
        self.deleted_count=0
        self.ALLOWED_SUFFIXES=['.jpg','jpeg','.png','.gif','.bmp','.tif','.tiff']
    def check_file(self,file_path):
        if file_path.suffix.lower() not in self.ALLOWED_SUFFIXES:
            return
        self.total_checked+=1
        img=None
        try:
            img=Image.open(file_path)
            img.verify()
        except(IOError,UnidentifiedImageError,OSError)as e:
            error_message=f"[损坏]文件：{file_path},错误：{e}"
            with open(self.log_file,'a')as f:
                f.write(error_message+"\n")
            if self.DELETE_FILES:
                try:
                    os.remove(file_path)
                    self.deleted_count += 1
                    print(f"->已删除文件：{file_path}")
                except Exception as del_e:
                    print(f"->无法删除文件:{file_path},错误：{del_e}")
        finally:
            if img is not None:
                img.close()
    def run_cleaning(self):
        print(f"---开始检查目录：{self.DATA_ROOT}---")
        for file_path in self.DATA_ROOT.rglob("*"):
            if file_path.is_file():
                self.check_file(file_path)
        print("---检查完毕---")
        print(f"总共检查文件数：{self.total_checked}")
        print(f"损坏/异常文件数：{self.deleted_count}")
        print(f"日志文件保存到：{self.log_file}")
#对图像数据的预处理
class ImagePreprocessor:
    def __init__(self,crop_size=(224,224),rotate_range=30,flip_prob=0.5):
        self.crop_size=crop_size
        self.rotate_range=rotate_range
        self.flip_prob=flip_prob
    def get_transform(self,is_train):
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.crop_size),
                transforms.RandomRotation(self.rotate_range),
                transforms.RandomHorizontalFlip(self.flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            resize_size=int(max(self.crop_size)/0.875)
            return transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
#划分数据集并导入
class CustomImageSplitter:
    def __init__(self,root_dir,preprocessor,batch_size=32):
        self.root_dir=root_dir
        self.preprocessor=preprocessor
        self.batch_size=batch_size
        self.load_pre_split_data()
    def load_pre_split_data(self):
        train_dir=os.path.join(self.root_dir,'train')
        test_dir=os.path.join(self.root_dir,'test')
        train_transform=self.preprocessor.get_transform(is_train=True)
        test_transform=self.preprocessor.get_transform(is_train=False)
        self.train_dataset=datasets.ImageFolder(
            root=train_dir,
            transform=train_transform
        )
        self.test_dataset=datasets.ImageFolder(
            root=test_dir,
            transform=test_transform
        )
        self.classes=self.train_dataset.classes

        print(f"数据加载完成。类别数：{len(self.classes)}")
        print(f"训练集大小：{len(self.train_dataset)},测试集大小:{len(self.test_dataset)}")
    def get_dataloaders(self):
        train_loader=DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=2)
        test_loader=DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=2)
        return train_loader,test_loader
#建立模型
class My_model(nn.Module):
    def __init__(self):
        super(My_model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(2,2)
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.dropout=nn.Dropout(0.2)
        self.fc=nn.Linear(in_features=128,out_features=5)
    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=self.pool(F.relu(self.bn3(self.conv3(x))))
        x=self.global_pool(F.relu(self.bn4(self.conv4(x))))
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.fc(x)
        return x
#运行代码
if __name__=="__main__":
    #数据清洗
    DATA_PATH=Path(r'C:\Users\31575\Desktop\flower')
    cleaner=DataCleaner(data_root_path=DATA_PATH,delete_enabled=False)
    cleaner.run_cleaning()
    #数据预处理
    preprocessor=ImagePreprocessor()
    #导入数据
    splitter=CustomImageSplitter(root_dir=DATA_PATH,preprocessor=preprocessor,batch_size=32)
    train_loader,test_loader=splitter.get_dataloaders()
    #模型训练
    model=My_model()
    train_accuracies=[]
    val_accuracies=[]
    device=torch.device('cpu')
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)
    num_epochs=10
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        correct_train, total_train = 0, 0
        for images, label in train_loader:
            device=torch.device('cpu')
            model.to(device)
            images=images.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            output=model(images)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _,predicted=torch.max(output,1)
            correct_train+=predicted.eq(label).sum().item()
            total_train+=label.size(0)
        train_accuracy=100*correct_train/total_train
        train_accuracies.append(train_accuracy)
        print(f"[Epoch{epoch+1}/{num_epochs}] loss: {running_loss/len(train_loader):.4f}")
        model.eval()
        correct_val,total_val=0,0
        with torch.no_grad():
            for images, label in test_loader:
                images=images.to(device)
                label=label.to(device)
                output=model(images)
                _,predicted=torch.max(output,1)
                total_val+=label.size(0)
                correct_val+=predicted.eq(label).sum().item()
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        print(f"Epoch{epoch+1}/{num_epochs},Train Accuracy:{train_accuracy:.2f}%,Val Accuracy: {val_accuracy:.2f}%")
    # 绘制准确率的图
    plt.figure(figsize=(10,5))
    plt.plot(range(1,num_epochs+1),train_accuracies,label='Train Accuracy',marker='o')
    plt.plot(range(1,num_epochs+1),val_accuracies,label='Validation Accuracy',marker='x')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.grid(True)
    plt.show()










