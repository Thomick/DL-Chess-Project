# Transform the model in a chess engine
import chess
import chess.pgn
import chess.engine
import numpy as np
import sys
from data_generator import encode_board
import torch

class ChessEngine():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_next_move(self, board):
        moves = board.legal_moves
        encodings = []
        for m in moves:
            board.push(m)
            encodings.append(encode_board(board))
            board.pop()
        encodings = torch.tensor(encodings,device = self.device)
        scores = self.model(encodings)
        chosen_move = moves[torch.argmax(scores)]
        return chosen_move

