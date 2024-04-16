import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class MLP_Classifier(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super().__init__()

        self.input_fc = nn.Linear(input_dimension, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dimension)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        y_predication = self.output_fc(h_2)

        return y_predication, h_2


INPUT_DIMENSION = 150 * 150
OUTPUT_DIMENSION = 25

model = MLP_Classifier(INPUT_DIMENSION, OUTPUT_DIMENSION)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_predication, _ = model(x)
        loss = criterion(y_predication, y)
        accuracy = calculate_accuracy(y, y_predication)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def calculate_accuracy(y, y_predication):
    top_predication = y_predication.argmax(1, keepdim=True)
    correct = top_predication.eq(y.view_as(top_predication)).sum()
    accuracy = correct.float() / y.shape[0]
    return accuracy


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_predication, _ = model(x)
            loss = criterion(y_predication, y)
            accuracy = calculate_accuracy(y, y_predication)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - (elapsed_minutes * 60))
    return elapsed_minutes, elapsed_seconds


train_image_data = np.load("Train_Images.npy")
train_label_data = np.load("Train_Labels.npy")

dummy_image_data = torch.from_numpy(train_image_data).unsqueeze(1).float()
dummy_label_data = torch.from_numpy(train_label_data).long()

if torch.cuda.is_available():
    dummy_image_data = dummy_image_data.cuda()
    dummy_label_data = dummy_label_data.cuda()

mean = dummy_image_data.float().mean() / 255
std = dummy_image_data.float().std() / 255

dummy_data_inter = dummy_image_data / 255
dummy_image_data = transforms.functional.normalize(dummy_data_inter, mean, std)

train_iterator = [
    (dummy_image_data[x * 44 : (x + 1) * 44], dummy_label_data[x * 44 : (x + 1) * 44])
    for x in range(int(len(dummy_image_data) / 44))
]

train_iterator = train_iterator[:-9]
valid_iterator = train_iterator[-9:-5]
test_iterator = train_iterator[-5:]

EPOCHS = 20

best_validation_loss = float("inf")

for epoch in range(EPOCHS):
    start_time = time.monotonic()
    train_loss, train_accuracy = train(model, train_iterator, optimizer, criterion, device)
    validation_loss, valid_accuracy = evaluate(model, valid_iterator, criterion, device)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), "baseline-model.pt")

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"\t Valid Loss: {validation_loss:.3f} |  Valid Accuracy: {valid_accuracy*100:.2f}%")


model.load_state_dict(torch.load("baseline-model.pt"))

test_loss, test_accuracy = evaluate(model, test_iterator, criterion, device)


print(f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_accuracy*100:.2f}%")