import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import GMChessDataset
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=774, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(in_features=2048, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, dataloader, size, epochs=1):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            targets = torch.unsqueeze(target.to(device).float(), 1)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.data.item()
        epoch_loss = running_loss/size
        print(f'Epoch {epoch} Loss: {epoch_loss}')
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            np.savetxt(f"../saves/loss{epoch}", np.array(losses))
            torch.save(model.state_dict(), f"../saves/sd{epoch}")
    return losses


def merge_datasets(datasets):
    finalds = datasets[0]
    for ds in datasets[1:]:
        finalds.merge(ds)
    return finalds

if __name__ == '__main__':
    batch_size = 256

    if len(sys.argv) < 2:
        print("Please provide datasets")
        sys.exit()

    datasets_path = sys.argv[1:]

    datasets = []
    for ds_path in datasets_path:
        datasets.append(torch.load(ds_path))
    final_dataset = merge_datasets(datasets)

    model = Net()
    dataloader = torch.utils.data.DataLoader(
        final_dataset, batch_size=batch_size, shuffle=True)
    losses = train_model(model, dataloader, len(dataset), epochs=100)
    plt.plot(losses)
    plt.show()
