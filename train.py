import json
from flask_server.university.nlp_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neural_net import NeuralNet

# Load intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

all_words = []
tags = []
xy = []
puncts = ['?', '!', '.', ',']

# Tokenize and collect all words and tags
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and remove punctuation
all_words = [stem(w) for w in all_words if w not in puncts]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

# Create training data
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Dataset
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Data loader
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

# Model config
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(1000):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device).long()

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')
