import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)

# Create a synthetic dataset and dataloaders
input_size = 1000
batch_size = 64
data_size = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.randn(data_size, input_size).to(device)
y = torch.randn(data_size, input_size).to(device)

train_data = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the model, loss function, and optimizer
model = SimpleModel().to(device)

# Use DataParallel to utilize multiple GPUs
if torch.cuda.device_count() > 1:
    print("Training on {} GPUs...".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 200

start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / (i+1)}")

end_time = time.time()
print("Training finished in {:.2f} seconds.".format(end_time - start_time))
