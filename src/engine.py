# Transform the model in a chess engine
import chess
import chess.pgn
import chess.engine
import numpy as np
import os
from data_generator import encode_board
import torch

stream = os.popen("which stockfish")

engine_path = stream.read().split('\n')[0]


class ChessEngine:
    def __init__(self):
        pass
    
    def play(self,board):
        board.push(board.legal_moves[0])

class TorchEngine(ChessEngine):
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

    def play(self, board):
        board.push(self.get_next_move(board))

class StockfishEngine(ChessEngine):
    def __init__(self, depth):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(depth=depth)
    
    def play(self,board):
        self.engine.play(board,self.limit)
    