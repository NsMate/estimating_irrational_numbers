import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Hyper parameters
num_classes = 10
batch_size = 1
learning_rate = 0.001

input_size = 10
sequence_length = 1
num_layers = 1

# Checking if there is a gpu to speed learning up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get data from data file
def get_data_from_file():
    f = open("./data/golden_r.txt", "r")
    return f.read()

def create_vectors_from_numbers(numbers):
    return_vectors = []
    for i in range(len(numbers) - 9):
        vector = []
        for j in range(10):
            vector.append(numbers[i + j])
        return_vectors.append(vector)
    return return_vectors

def create_train_and_test_loaders_by_size(size):
    data = get_data_from_file()

    learn_data = data[:size]
    test_data = data

    # Input and target arrays
    learn_vectors = []
    test_vectors = []

    learn_labels = []
    test_labels = []

    learn_data = [int(i) for i in str(learn_data)]
    test_data = [int(i) for i in str(test_data)]

    learn_labels = np.array(learn_data[9:], dtype=float)
    test_labels = np.array(test_data[9:], dtype=float)

    learn_vectors = create_vectors_from_numbers(learn_data)
    test_vectors = create_vectors_from_numbers(test_data)

    learn_vectors = np.array(learn_vectors, dtype=float)
    test_vectors = np.array(test_vectors, dtype=float)

    # Creating datasets from the data
    train_set = TensorDataset(torch.from_numpy(learn_vectors), torch.from_numpy(learn_labels))
    test_set = TensorDataset(torch.from_numpy(test_vectors), torch.from_numpy(test_labels))

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)

    return (train_loader, test_loader)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = x.view(batch_size, sequence_length, input_size)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.fc(out)
        return out

for i in range(0, 9, 1):
    hidden_size = pow(2, i)

    for j in range(2, 7):
        set_sizes = pow(10, j)

        loaders = create_train_and_test_loaders_by_size(set_sizes)

        train_loader = loaders[0]
        test_loader = loaders[1]

        model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        n_total_steps = len(train_loader)
        loss_avg = 100
        number_epochs = 0
        while loss_avg > 0.7:
            loss_avg = 0
            number_epochs = number_epochs + 1
            for i, (numbers, labels) in enumerate(train_loader):

                numbers = numbers.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(numbers.float())
                loss = criterion(outputs, labels.long())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_avg = loss_avg + loss.item()
            
            loss_avg = loss_avg / n_total_steps

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for numbers, labels in test_loader:
                numbers = numbers.to(device)
                labels = labels.to(device)

                outputs = model(numbers.float())
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network, using LSTM model with ' + str(hidden_size) +  ' size hidden dimension and ' + str(set_sizes) + ' train set size: ' + str(acc) + ' %, Epochs: ' + str(number_epochs))