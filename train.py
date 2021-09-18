import json
from nltk_utils import tokenize, bag_of_words, stem
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:

    # save the tags in a list
    tag = intent['tag']
    tags.append(tag)

    # save the patterns in a list
    # create a tuple of tags and the words in a pattern
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted((set(tags)))

x_train = []
y_train = []

# create bag of words for tags and the tokenized sentences against them
for(pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    # convert tags as index labels and append
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Define DataLoader

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    #dataset[idx]
    #iterate data through indexes
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyper parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
print(input_size, len(all_words))
print(output_size, len(tags))
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
for epoch in range(num_epochs):

    for(words, labels) in train_loader:

        # send words and labels to device
        words = words.to(device)
        labels = labels.type(torch.LongTensor).to(device)

        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backwards pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'Final loss = {loss.item():.4f}')

data = {
            'model_state': model.state_dict(),
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size,
            'all_words': all_words,
            'tags': tags
        }

FILE = 'data.pth'
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')

