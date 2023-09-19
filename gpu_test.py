import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)

# Create a PyTorch Lightning module for training
class SimpleLightningModule(pl.LightningModule):
    def __init__(self, batch_size, lr):
        super(SimpleLightningModule, self).__init__()
        self.model = SimpleModel()
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.MSELoss()(outputs, y)
        
        # Log the loss for printing
        self.log('train_loss', loss)
        
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)

# Create a synthetic dataset
input_size = 1000
batch_size = 64
data_size = 10000

x = torch.randn(data_size, input_size)
y = torch.randn(data_size, input_size)

train_data = TensorDataset(x, y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the LightningModule
lr = 0.01
lightning_module = SimpleLightningModule(batch_size=batch_size, lr=lr)

# Initialize the Trainer for multi-GPU training
trainer = pl.Trainer(devices=torch.cuda.device_count() if torch.cuda.is_available() else 0, log_every_n_steps=5, max_epochs=100)

# Start training
trainer.fit(lightning_module, train_loader)
