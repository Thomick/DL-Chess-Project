import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import ChessDataset
import sys
from torch.utils.data import Dataset
from progressbar import progressbar
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=774, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(in_features=2048, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)


def save(model, losses, val_losses, identifier):
    np.savetxt(f"../saves/mlp/eval_loss{identifier}.txt", np.array(val_losses))
    np.savetxt(f"../saves/mlp/loss{identifier}.txt", np.array(losses))
    torch.save(model.state_dict(), f"../saves/mlp/sd{identifier}.pt")
    print("Progress saved !")


def eval_model(model, val_loader, size):
    print("Evaluation ...")
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    for inputs, target in progressbar(val_loader):
        inputs = inputs.to(device)
        targets = torch.unsqueeze(target.to(device).float(), 1)
        outputs = model(inputs.float())
        loss = criterion(outputs, targets)
        running_loss += loss.data.item()
    return running_loss/size


def train_model(model, train_loader, val_loader, train_size, val_size, epochs=1):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)

    losses = []
    val_losses = []
    print(f"Training for {epochs} epochs ...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting epoch {epoch} ...")
        for inputs, target in progressbar(train_loader):
            inputs = inputs.to(device)
            targets = torch.unsqueeze(target.to(device).float(), 1)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.data.item()
        epoch_loss = running_loss/train_size
        print("Validation")
        val_loss = eval_model(model, val_loader, val_size)

        print(f'Epoch {epoch} Loss: {epoch_loss} Validation: {val_loss}')
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0:
            save(model, losses, val_losses, epoch)
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

    trainset_size = int(len(final_dataset)*0.8)
    valset_size = len(final_dataset)-trainset_size
    train_set, val_set = torch.utils.data.random_split(
        final_dataset, [trainset_size, valset_size])

    print("Size of the dataset : ", len(final_dataset))
    model = Net().to(device)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True)
    losses, val_losses = train_model(
        model, train_loader, val_loader, trainset_size, valset_size, epochs=1000)
    save(model, losses, "final")
