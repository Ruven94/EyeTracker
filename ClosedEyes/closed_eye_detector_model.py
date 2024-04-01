### >>> Import ###
import random
from torch.utils.data import DataLoader, TensorDataset
from nn_closed_eyes_detection import *
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision.transforms as transforms
import time
import numpy as np
### <<< Import ###

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

print(f'Process started')

images = np.load('MRL_data_images.npy', allow_pickle= True)
labels = np.load('MRL_data_labels.npy', allow_pickle= True)

combined_data = list(zip(images, labels))
random.shuffle(combined_data)
images, labels = zip(*combined_data)

split_index = int(0.8 * len(images))
train_images, test_images = images[:split_index], images[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

transform = transforms.Compose([transforms.ToTensor()])
train_images_tensor = torch.stack([transform(img) for img in train_images])
test_images_tensor = torch.stack([transform(img) for img in test_images])

train_labels_tensor = torch.LongTensor(train_labels)
test_labels_tensor = torch.LongTensor(test_labels)

trainset = TensorDataset(train_images_tensor, train_labels_tensor)
testset = TensorDataset(test_images_tensor, test_labels_tensor)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False)

print(f'Data loading complete')

model = ClosedEyeDetection().to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.95)

epochs = 300
accuracy_list = []
loss_list = []
for epoch in range(1,epochs + 1):
    start_time = time.time()
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch}/{epochs} | Time: {time.time() - start_time:.2f} | TrainLoss: {loss.item():.2f} | Validation Accuracy: {accuracy:.3f}')
    accuracy_list.append(accuracy)
    loss_list.append(loss.item())

    if epoch%100 == 0:
        torch.save({'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict()
                      }, f'Closed_model_{epoch}.pt')
        np.save('model_accuracy.npy', accuracy_list)
        np.save('model_loss.npy', loss_list)
