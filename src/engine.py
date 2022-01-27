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


class ChessEngine:
    def __init__(self):
        pass

    def play(self, board, color):
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
        scores = self.model(encodings.float()) * color
        
        scores = scores.cpu().detach().numpy().flatten()
        n = max(int(0.25*scores.shape[0]), 1)
        #print(scores)
        #print(scores.shape)
        possible_moves = np.argpartition(scores, -n)[-n:]
        #print(possible_moves)
        #print(possible_moves.shape)
        chosen_move = moves[0]
        if possible_moves.shape[0] != 1:
            weights = scores[possible_moves]
            weights = weights - min(weights)
            weights = weights/np.sum(weights)
            move_id = np.random.choice(possible_moves.squeeze(), p=weights)
            chosen_move = moves[move_id]
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
            translated = mlp_encoding_to_cnn_encoding_board_only(encode_board(board))
            encodings.append(translated )
            board.pop()
        
        encodings = np.array(encodings)
        n = encodings.shape[0]
        boards = np.concatenate(encodings[:,0],axis=0).reshape(n,12,8,8)
        metas = np.concatenate(encodings[:,1],axis=0).reshape(n,6)
        boards = torch.tensor(boards,device = self.device)
        metas = torch.tensor(metas,device = self.device)
        scores = self.model((boards.float(),metas.float())) * color

        scores = scores.cpu().detach().numpy().flatten()
        n = max(int(0.25*scores.shape[0]), 1)
        #print(scores)
        #print(scores.shape)
        possible_moves = np.argpartition(scores, -n)[-n:]
        #print(possible_moves)
        #print(possible_moves.shape)
        chosen_move = moves[0]
        if possible_moves.shape[0] != 1:
            weights = scores[possible_moves]
            weights = weights - min(weights)
            weights = weights/np.sum(weights)
            move_id = np.random.choice(possible_moves.squeeze(), p=weights)
            chosen_move = moves[move_id]
        return chosen_move

    def play(self, board, color):
        board.push(self.get_next_move(board, color))

class CNNalphabeta_Engine(ChessEngine):
    def __init__(self, model, device, depth):
        self.model = model
        self.device = device
        self.depth = depth

    # color = -1 when black, and = 1 when white
    def evaluate(self, board):
        translated = mlp_encoding_to_cnn_encoding_board_only(encode_board(board))
        n = translated.shape[0]
        board = translated[0]
        meta = translated[1]

        board = torch.tensor(board,device = self.device)
        board = torch.unsqueeze(board, dim=0)

        meta = torch.tensor(meta,device = self.device)
        meta = torch.unsqueeze(meta, dim=0)

        score = self.model((board.float(),meta.float()))

        return score.cpu().detach().numpy()[0,0]

    def play(self, board, color):
        move = alphaBetaRoot(board,self.evaluate, color, self.depth)
        board.push(move)


class StockfishEngine(ChessEngine):
    def __init__(self, time, depth):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(time=time, depth=depth)

    def play(self, board, color):
        result = self.engine.play(board, self.limit)
        board.push(result.move)

    def quit(self):
        self.engine.quit()


def reset_cursor():
    BEGIN = "\033[F"
    UP = "\033[A"
    print(UP*8 + BEGIN)


def play_game(engine1, engine2, out, max_length=500):
    board = chess.Board()

    if out:
        print(board)
    for i in range(max_length):
        #print("Turn ",i)
        start = time.time()
        engine1.play(board, 1)
        end = time.time()
        if out:
            reset_cursor()
            print(board)
        if board.is_game_over():
            break
        start = time.time()
        engine2.play(board, -1)
        end = time.time()
        if out:
            reset_cursor()
            print(board)
        if board.is_game_over():
            break
    outcome = board.outcome()
    print(chess.pgn.Game().from_board(board))
    if outcome == None:
        print("Stopping the game after", max_length, "steps")
        return 1.0/2.0, 1.0/2.0
    print(outcome.termination)

    scores = outcome.result()
    if scores == "1/2-1/2":
        return 1.0/2.0, 1.0/2.0
    scores = scores.split("-")
    return int(scores[0]), int(scores[1])


def compare_engines(n, engine1, engine2, display):
    wins1_white = 0
    wins2_white = 0
    wins1_black = 0
    wins2_black = 0

    draws = 0

    for x in range(n):
        res1, res2 = play_game(engine1, engine2, display)

        if res1 == res2:
            draws += 1
        else:
            wins1_white += res1
            wins2_black += res2
        print("Game {} finished".format(1+x))

    for x in range(n):
        res2, res1 = play_game(engine2, engine1, display)

        if res1 == res2:
            draws += 1
        else:
            wins1_black += res1
            wins2_white += res2

        print("Game {} finished".format(1+x+n))

    f = open("results_vs.txt", "w")
    print("Engine 1 wins as white = {} || Engine 1 wins as black = {} || Engine 1 total wins = {}".format(wins1_white, wins1_black,wins1_white+wins1_black ))
    print("Number of draws = {}".format(draws ))
    print("Engine 2 wins as white = {} || Engine 2 wins as black = {} || Engine 2 total wins = {}".format(wins2_white, wins2_black,wins2_white+wins2_black ))

    f.write("Engine 1 wins as white = {} || Engine 1 wins as black = {} || Engine 1 total wins = {}\n".format(wins1_white, wins1_black,wins1_white+wins1_black ))
    f.write("Number of draws = {}\n".format(draws ))
    f.write("Engine 2 wins as white = {} || Engine 2 wins as black = {} || Engine 2 total wins = {}\n".format(wins2_white, wins2_black,wins2_white+wins2_black ))


if __name__ == "__main__":
    random_engine = ChessEngine()
    stock = StockfishEngine(time=2, depth=1)
    stock2 = StockfishEngine(time=2, depth=11)
    cnn = CNN_Net()
    cnn.load_state_dict(torch.load("../saves/cnn/cnnfinal.pt"))
    cnn.eval()
    mlp = Net()
    mlp.load_state_dict(torch.load("../saves/mlp/sdfinal.pt"))
    mlp.eval()
    cnn_engine = CNN_Engine(cnn.to(device), device)
    mlp_engine = TorchEngine(mlp.to(device), device)
    #cnn_engine_ab = CNNalphabeta_Engine(cnn.to(device), device, 4)
    compare_engines(1000,cnn_engine, mlp_engine, False)
    stock.quit()
    stock2.quit()
    random_engine.quit()
