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
    
    def play(self,board, color):
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
    def __init__(self, time, depth):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(time=time, depth=depth)
    
    def play(self,board, color):
        result = self.engine.play(board,self.limit)
        board.push(result.move)

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

def reset_cursor():
    BEGIN = "\033[F"
    UP = "\033[A"
    print(UP*8 + BEGIN)

def play_game(engine1, engine2, out, max_length = 500):
    board = chess.Board()

    if out:
        print(board)
    for i in range(max_length):
        #print("Turn ",i)
        engine1.play(board, 1)
        if out:
            reset_cursor()
            print(board)
        if board.is_game_over():
            break
        engine2.play(board, -1)
        if out:
            reset_cursor()
            print(board)
        if board.is_game_over():
            break
    outcome = board.outcome()
    print(board_to_game(board))

    print(chess.pgn.Game().from_board(board))
    if outcome == None:
        print("Stopping the game after", max_length, "steps")
        return 1/2, 1/2
    print(outcome.termination)

    scores = outcome.result()
    if scores == "1/2-1/2":
        return 1/2, 1/2
    scores = scores.split("-")
    return int(scores[0]), int(scores[1])

def compare_engines(n,engine1,engine2, display):
    score1 = 0
    score2 = 0

    for x in range(n):
        res1, res2 = play_game(engine1, engine2, display)
        score1 += res1
        score2 += res2
        print("Game {} finished".format(1+x))

    for x in range(n):
        res2, res1 = play_game(engine2, engine1, display)
        score1 += res1
        score2 += res2
        print("Game {} finished".format(1+x+n))

    print("Engine 1 score = {} || Engine 2 score = {}".format(score1, score2))


if __name__ == "__main__":
    random_engine = ChessEngine()
    stock = StockfishEngine(time=2, depth=12)
    stock2 = StockfishEngine(time=2, depth=11)
    cnn = CNN_Engine(CNN_Net().to(device), device)
    compare_engines(10,stock, stock2, True)
    stock.quit()
    stock2.quit()
    random_engine.quit()
