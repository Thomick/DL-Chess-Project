import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        self.layers = nn.Sequential(
            nn.Linear(in_features=774, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ELU(),
            #nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, dataloader, batch_size, epochs=1):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            epoch_loss += loss.data.item()
        print(f'Epoch {epoch} Loss: {epoch_loss}')
        losses.append(epoch_loss)


if __name__ == '__main__':
    model = Net()
