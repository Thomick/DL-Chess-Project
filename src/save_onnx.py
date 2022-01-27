# Transform the model in a chess engine
import time
import chess
import chess.pgn
import chess.engine
import numpy as np
import random
import os
from data_generator import encode_board
import torch
from alphabeta import alphaBetaRoot
from alphabeta import alphaBetaRoot_mt
import collections
from cnn import CNN_Net
from mlp import Net
from cnn import mlp_encoding_to_cnn_encoding_board_only

stream = os.popen("which stockfish")

engine_path = stream.read().split('\n')[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    cnn = CNN_Net()
    cnn.load_state_dict(torch.load("../saves/cnn/cnnfinal.pt"))
    cnn.eval()
    cnn.train(False)

    dummy_1 = torch.randn(1, list([1,12,8,8]), requires_grad=True) 
    dummy_2 = torch.randn(1, list([1,6]), requires_grad=True) 

    data = (dummy_1, dummy_2)

    torch.onnx.export(cnn,               # model being run
                  data,                         # model input (or a tuple for multiple inputs)
                  "cnn.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=False,        # store the trained parameter weights inside the model file
                  opset_version=10)          # the ONNX version to export the model to
