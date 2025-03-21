import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载NPZ数据
def load_npz_data(npz_file):
    data = np.load(npz_file)
    return data['data'], data['labels']

# 自定义数据集类
class CustomAudioDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        self.data, self.labels = load_npz_data(npz_file)
        self.transform = transform
        
        # 构建标签到索引的映射
        unique_labels = list(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取原始数据
        spectrogram = self.data[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量并调整维度
        if self.transform:
            transformed = self.transform(spectrogram)
        else:
            transformed = torch.FloatTensor(spectrogram)
        
        # 强制确保形状为 [C, H, W]
        if transformed.dim() == 2:
            transformed = transformed.unsqueeze(0)
        elif transformed.dim() == 3:
            # 若输入是 [H, W, C]，调整为 [C, H, W]
            transformed = transformed.permute(2, 0, 1)
        
        label_idx = self.label_to_idx[label]
        return transformed, torch.tensor(label_idx, dtype=torch.long)

class CoAtNet(nn.Module):
    def __init__(self, num_classes=36):
        super(CoAtNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 初始卷积层
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 新增的卷积层
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 增加通道数的卷积层
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)  # 另一个池化层
        )
        
        # 使用一个示例输入来动态计算全连接层的输入尺寸
        example_input = torch.randn(1, 1, 64, 156)  # 假设输入图像大小为64x156
        example_output = self.conv_layers(example_input)
        self.fc_input_size = example_output.view(1, -1).size(1)
        
        self.fc = nn.Linear(self.fc_input_size, num_classes)  # 输出层
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


# 训练函数
def train():
    npz_file = 'spectrogram_data.npz'
    dataset = CustomAudioDataset(
        npz_file,
        transform=Compose([
            ToTensor(),  # 转换为 [C, H, W]
            Lambda(lambda x: x.permute(2, 0, 1) if x.shape[0] != 1 else x)  # 强制通道在第一位
        ])
    )
    
    # 打印验证数据形状
    sample, _ = dataset[0]
    print(f"验证数据形状: {sample.shape}")  # 应输出 torch.Size([1, 64, 156])
    
    # 处理标签
    labels = [dataset.label_to_idx[label] for label in dataset.labels]
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, stratify=labels
    )
    
    # 创建采样器和数据加载器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)
    
    # 初始化模型
    model = CoAtNet(num_classes=len(dataset.label_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 1100
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()

        
        # 验证阶段
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            # with tqdm(total=len(val_loader), desc="Validation") as pbar_val:
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                    
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"{epoch+1},Validation Accuracy: {accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'Mymodel.pt')

if __name__ == "__main__":
    train()