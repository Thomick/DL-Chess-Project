import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import ChessDataset
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset
from progressbar import progressbar

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


def save(model, losses, identifier):
    np.savetxt(f"../saves/loss{identifier}", np.array(losses))
    torch.save(model.state_dict(), f"../saves/sd{identifier}")


def train_model(model, dataloader, size, epochs=1):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    print(f"Training for {epochs} epochs ...")

    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Starting epoch {epoch} ...")
        for inputs, target in progressbar(dataloader):
            inputs = inputs.to(device)
            targets = torch.unsqueeze(target.to(device).float(), 1)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.data.item()/size
        epoch_loss = running_loss
        print(f'Epoch {epoch} Loss: {epoch_loss}')
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            save(model, losses, epoch)
    return losses


class FixedChessDataset(Dataset):
    def __init__(self, datasets):
        self.encodings = []
        for ds in datasets:
            self.encodings += [l[1] for l in ds.encodings.items()]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        sample = (self.encodings[idx][0], self.encodings[idx][1])
        return sample

if __name__ == '__main__':
    batch_size = 256

    if len(sys.argv) < 2:
        print("Please provide datasets")
        sys.exit()

    datasets_path = sys.argv[1:]

    datasets = []
    for ds_path in datasets_path:
        datasets.append(torch.load(ds_path))
    final_dataset = FixedChessDataset(datasets)

    print("Size of the dataset : ", len(final_dataset))
    model = Net()
    dataloader = torch.utils.data.DataLoader(
        final_dataset, batch_size=batch_size, shuffle=True)
    losses = train_model(model, dataloader, len(final_dataset), epochs=1000)
    save(model, losses, "final")
    plt.plot(losses)
    plt.show()
