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

# Define the CNN architecture
class LightweightMNISTCNN(nn.Module):
    def __init__(self):
        super(LightweightMNISTCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 24),
            nn.ReLU(),
            nn.Linear(24, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = LightweightMNISTCNN().to(device)
    
    # Print model summary
    print("\nModel Summary:")
    print("-------------")
    model_summary = summary(model, input_size=(1, 28, 28))
    print(model_summary)
    print("\n")

    # Enhanced data augmentation pipeline with v2 transforms
    train_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets with size specifications
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Use only 50000 images for training
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, 
        [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LightweightMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {param_count}")

    # Training loop with epochs
    print("Starting training...")
    num_epochs = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar description with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print epoch average loss
        epoch_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}/{num_epochs} complete. Average loss: {epoch_loss:.4f}')
        
        # Evaluate after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # Wrap test_loader with tqdm
            test_pbar = tqdm(test_loader, desc='Evaluating')
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Update progress bar with current accuracy
                current_acc = 100 * correct / total
                test_pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%\n')

    print("Training completed!")
    print(f'Final Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    train_model() 


