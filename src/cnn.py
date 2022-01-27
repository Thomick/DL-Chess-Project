import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generator import ChessDataset
from mlp import FixedChessDataset
import matplotlib.pyplot as plt
import sys
import numpy as np
from torch.utils.data import Dataset
from progressbar import progressbar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(12,128,5)
        self.conv2 = nn.Conv2d(128,256,3)
        self.fc1 = nn.Linear(1030,2048)
        self.fc2 = nn.Linear(2048,1)

    def forward(self, x):
        board, meta = x

        board = self.conv1(board)
        board = F.elu(board)
        board = self.conv2(board)
        board = F.elu(board)

        board = torch.flatten(board,1)

        data = torch.cat((board,meta), dim=1)

        res = self.fc1(data)
        res = F.elu(res)
        res = self.fc2(res)

        return res


def save(model, losses, val_losses, identifier):
    np.savetxt(f"../saves/cnn/eval_loss{identifier}.txt", np.array(val_losses))
    np.savetxt(f"../saves/cnn/loss{identifier}.txt", np.array(losses))
    torch.save(model.state_dict(), f"../saves/cnn/cnn{identifier}.pt")
    print("Progress saved !")

def eval_model(model, val_loader, size):
    print("Evaluation ...")
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    for inputs, target in progressbar(val_loader):
        boards = inputs[0]
        metas = inputs[1]
        boards = boards.to(device)
        metas = metas.to(device)
        targets = torch.unsqueeze(target.to(device).float(), 1)
        outputs = model((boards.float(),metas.float()))
        loss = criterion(outputs, targets)
        running_loss += loss.data.item()
    return running_loss/size

def train_model(model, train_loader, val_loader, train_size, val_size, epochs=1):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    losses = []
    val_losses = []
    print(f"Training for {epochs} epochs ...")

    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Starting epoch {epoch} ...")
        for inputs, target in progressbar(train_loader):
            boards = inputs[0]
            metas = inputs[1]
            boards = boards.to(device)
            metas = metas.to(device)
            targets = torch.unsqueeze(target.to(device).float(), 1)
            outputs = model((boards.float(),metas.float()))
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.data.item()/train_size
        epoch_loss = running_loss
        print("Validation")
        val_loss = eval_model(model, val_loader, val_size)

        print(f'Epoch {epoch} Loss: {epoch_loss} Validation: {val_loss}')
        losses.append(epoch_loss)
        if epoch % 100 == 0:
            save(model, losses, val_losses, epoch)
    return losses, val_losses

def mlp_encoding_to_cnn_encoding_board_only(encoding):
    wpawn = encoding[0:64].reshape((8,8))
    wbishop = encoding[64:128].reshape((8,8))
    wknight = encoding[128:192].reshape((8,8))
    wrook = encoding[192:256].reshape((8,8))
    wqueen = encoding[256:320].reshape((8,8))
    wking = encoding[320:384].reshape((8,8))
    bpawn = encoding[384:448].reshape((8,8))
    bbishop = encoding[448:512].reshape((8,8))
    bknight = encoding[512:576].reshape((8,8))
    brook = encoding[576:640].reshape((8,8))
    bqueen = encoding[640:704].reshape((8,8))
    bking = encoding[704:768].reshape((8,8))
    meta = np.array(encoding[768:])

    board = np.array([wpawn, wbishop,wknight,wrook, wqueen, wking,\
            bpawn, bbishop,bknight,brook, bqueen, bking])

    return np.array([board,meta], dtype=object) 


def mlp_encoding_to_cnn_encoding(encoding):
    wpawn = encoding[0][0:64].reshape((8,8))
    wbishop = encoding[0][64:128].reshape((8,8))
    wknight = encoding[0][128:192].reshape((8,8))
    wrook = encoding[0][192:256].reshape((8,8))
    wqueen = encoding[0][256:320].reshape((8,8))
    wking = encoding[0][320:384].reshape((8,8))
    bpawn = encoding[0][384:448].reshape((8,8))
    bbishop = encoding[0][448:512].reshape((8,8))
    bknight = encoding[0][512:576].reshape((8,8))
    brook = encoding[0][576:640].reshape((8,8))
    bqueen = encoding[0][640:704].reshape((8,8))
    bking = encoding[0][704:768].reshape((8,8))
    meta = np.array(encoding[0][768:])

    board = np.array([wpawn, wbishop,wknight,wrook, wqueen, wking,\
            bpawn, bbishop,bknight,brook, bqueen, bking])

    return np.array([np.array([board,meta], dtype=object), encoding[1]], dtype=object) 

class CNNChessDataset(Dataset):
    def __init__(self, chess_dataset):
        self.encodings = []
        for e in chess_dataset.encodings:
            self.encodings += [ mlp_encoding_to_cnn_encoding(e)]
            

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        sample = ([self.encodings[idx][0][0],self.encodings[idx][0][1]], self.encodings[idx][1]) 
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

    cnn_dataset = CNNChessDataset(final_dataset)

    trainset_size = int(len(cnn_dataset)*0.95)
    valset_size = len(cnn_dataset)-trainset_size
    train_set, val_set = torch.utils.data.random_split(
        cnn_dataset, [trainset_size, valset_size])

    print("Size of the training dataset : ", len(train_set))
    model = CNN_Net().to(device)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True)
    losses, val_losses = train_model(
        model, train_loader, val_loader, trainset_size, valset_size, epochs=1000)
    save(model, losses, val_losses, "final")
