import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F
import random

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EfficientMNISTCNN(nn.Module):
    def __init__(self, dropout_rate=0.15):
        super(EfficientMNISTCNN, self).__init__()
        
        # First block - Initial feature extraction
        # Using stride 1 here to preserve early features
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2),
        )
        
        # Second block - Feature refinement with stride
        # Using stride 2 in conv to learn spatial downsampling
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16), 
            nn.Conv2d(16, 16, kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
        )
        
        # Third block - Deep features
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, kernel_size=3),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AvgPool2d(3,2)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model():
    # Set seed
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = EfficientMNISTCNN().to(device)
    
    # Print model summary
    print("\nModel Summary:")
    print("-------------")
    summary(model, input_size=(1, 28, 28))
    print("\n")

    # Enhanced data augmentation - slightly modified
    train_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomRotation(7),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    
    # Modified learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # Reduced max learning rate
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Longer warmup
        anneal_strategy='cos'
    )

    best_accuracy = 0
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        # Test phase at the end of each epoch
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_accuracy = 100 * correct / total
        print(f'\nEpoch {epoch+1}:')
        print(f'Test Accuracy: {test_accuracy:.2f}%')

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'\nBest Test Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    train_model()





