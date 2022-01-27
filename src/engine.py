# Transform the model in a chess engine
import chess
import chess.pgn
import chess.engine
import numpy as np
import random
import os
from data_generator import encode_board
import torch
import collections
from cnn import CNN_Net
from mlp import Net
from cnn import mlp_encoding_to_cnn_encoding_board_only

stream = os.popen("which stockfish")

engine_path = stream.read().split('\n')[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ChessEngine:
    def __init__(self):
        pass
    
    def play(self,board):
        board.push(random.choice(list(board.legal_moves)))
    
    def quit(self):
        pass

class TorchEngine(ChessEngine):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_next_move(self, board, color):
        moves = list(board.legal_moves)
        encodings = []
        for m in moves:
            board.push(m)
            encodings.append(encode_board(board))
            board.pop()

        encodings = torch.tensor(encodings,device = self.device)
        scores = self.model(encodings) * -1
        chosen_move = moves[torch.argmax(scores)]
        return chosen_move

    def play(self, board, color):
        board.push(self.get_next_move(board, color))


class CNN_Engine(ChessEngine):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    # color = -1 when black, and = 1 when white
    def get_next_move(self, board, color):
        moves = list(board.legal_moves)
        encodings = []
        for m in moves:
            board.push(m)
            encodings.append( mlp_encoding_to_cnn_encoding_board_only(encode_board(board)))
            board.pop()

        encodings = np.array(encodings)
        boards = encodings[0]
        metas = encodings[0]
        boards = torch.tensor(boards,device = self.device)
        metas = torch.tensor(metas,device = self.device)
        scores = self.model((boards.float(),metas.float())) * -1
        chosen_move = moves[torch.argmax(scores)]
        return chosen_move

    def play(self, board, color):
        board.push(self.get_next_move(board, color))

class StockfishEngine(ChessEngine):
    def __init__(self, time):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(time=time)
    
    def play(self,board, color):
        self.engine.play(board,self.limit)

    def quit(self):
        self.engine.quit()
    
def board_to_game(board):
    game = chess.pgn.Game()

    switchyard = collections.deque()
    while board.move_stack:
        switchyard.append(board.pop())

    game.setup(board)
    node = game

    while switchyard:
        move = switchyard.pop()
        node = node.add_variation(move)
        board.push(move)

    return game

def play_game(engine1, engine2, white, max_length = 500):
    board = chess.Board()
    white = engine1
    black = engine2
    if white != 1:
        white = engine2
        black = engine1

    for i in range(max_length):
        #print("Turn ",i)
        white.play(board)
        if board.is_game_over(claim_draw = True):
            break
        black.play(board)
        if board.is_game_over(claim_draw = True):
            break
    outcome = board.outcome()
    print(board_to_game(board))
    if outcome == None:
        print("Stopping the game after",max_length,"steps")
        return 1/2,1/2
    print(outcome.termination)
    scores = outcome.result()
    if scores == "1/2-1/2": 
        return 1/2,1/2
    return int(scores[1]),int(scores[2])

if __name__ == "__main__":
    random_engine = ChessEngine()
    stock = StockfishEngine(time=0.1)
    cnn = CNN_Engine(CNN_Net().to(device), device)
    scores = play_game(random_engine, stock)
    print(scores)
    stock.quit()
    random_engine.quit()
