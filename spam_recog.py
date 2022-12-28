import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M")

f = open("C:/Users/adenm/OneDrive/Desktop/Research/ChinwenduRG/Spam_recognition/reports/model_report_" + dt_string + ".txt", "w+")

#hyper parameters
batch_size = 128
num_epochs = 30 
learning_rate = 0.01
input_size = 57 
hidden_size = 250

f.write("batch_size = " + str(batch_size))
f.write(f"\nnum_epochs = " + str(num_epochs))
f.write(f"\nlearning_rate = " + str(learning_rate))
f.write(f"\ninput_size = " + str(input_size))
f.write(f"\nhidden_size = " + str(hidden_size) + "\n")

class NeuralNetowrk(torch.nn.Module):
    def __init__(self):
        super(NeuralNetowrk, self).__init__()
        self.neural_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size*2,hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size*2,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,1)
        )

    def forward(self, input):
        return self.neural_stack(input) 

dataset = pd.read_csv(r"C:/Users/adenm/OneDrive/Desktop/Research/ChinwenduRG/Spam_recognition/spambase.data", header=None)
columns = list(range(0, 58, 1))
columns = [str(x) for x in columns]
columns[-1] = 'labels'
dataset.columns = columns
labels = dataset['labels']
del dataset['labels']

data_train, data_test, labels_train, labels_test = train_test_split(dataset, labels, test_size = 0.3, random_state = 1, stratify = labels)

class BinaryDataset(Dataset):
    def __init__(self, data_array, label_array):
        self.data_array = data_array
        self.label_array = label_array

    def __getitem__(self, index):
        data_return = torch.tensor(self.data_array, dtype=torch.float32)
        label_return = torch.tensor(self.label_array, dtype=torch.float32)
        return data_return[index], label_return[index]

    def __len__(self):
        return len(self.data_array)


train_dataset = BinaryDataset(np.array(data_train), np.array(labels_train))
test_dataset = BinaryDataset(np.array(data_test), np.array(labels_test))

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = NeuralNetowrk()

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    correct = 0
    train_losses = []
    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data, labels.unsqueeze(dim=1)

        prediction = model(data)
        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        correct += (((torch.sigmoid(prediction) > 0.5)  * 1.0) == labels).sum()

        if batch % 100 == 0:
            print(f"Train loss: {loss.item()}")
            f.write(f"\nTrain loss: {loss.item()}")

    print(f"Train Accuracy: {correct/len(dataloader.dataset)}")
    f.write(f"\nTrain Accuracy: {correct/len(dataloader.dataset)}")
    return train_losses

def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data, labels.unsqueeze(dim=1)
            prediction = model(data)
            test_loss += loss_fn(prediction, labels).item()
            correct += (((torch.sigmoid(prediction) > 0.5)  * 1.0) == labels).sum()

    print(f"Test Average Loss: {test_loss/len(dataloader)}\nTest Accuracy: {correct/len(dataloader.dataset)}")
    f.write(f"\nTest Average Loss: {test_loss/len(dataloader)}\nTest Accuracy: {correct/len(dataloader.dataset)}\n")
    return test_loss

for t in range(num_epochs):
    print(f"\n-------------------------------\nEpoch {t+1}\n-------------------------------")
    f.write(f"\n-------------------------------\nEpoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")

f.close()
