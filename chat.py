import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_initialized() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load trained model details
FILE = 'data.pth'
data = torch.load(FILE)

# import the weights and structure of trained model from saved file
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

# define/load the NN structure
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Memaw'
print('''Let's chat! type quit to exit''')

while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # get probability of tags
    # if probability > 0.75 display a random response from the predicted tag
    probs = torch.softmax(output, dim =1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f'{bot_name}: I do not understand ....')
