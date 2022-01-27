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

<<<<<<< HEAD
        encodings = torch.tensor(encodings,device = self.device)
        scores = self.model(encodings) * color
=======
        encodings = torch.tensor(encodings, device=self.device)
        scores = self.model(encodings) * -1
>>>>>>> 743b195641a46de1a6d5c249c2b1e5c0bfe425e8
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
<<<<<<< HEAD
            translated = mlp_encoding_to_cnn_encoding_board_only(encode_board(board))
            encodings.append(translated )
=======
            encodings.append(
                mlp_encoding_to_cnn_encoding_board_only(encode_board(board)))
>>>>>>> 743b195641a46de1a6d5c249c2b1e5c0bfe425e8
            board.pop()
        
        encodings = np.array(encodings)
<<<<<<< HEAD
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
=======
        boards = encodings[0]
        metas = encodings[0]
        boards = torch.tensor(boards, device=self.device)
        metas = torch.tensor(metas, device=self.device)
        scores = self.model((boards.float(), metas.float())) * -1
        chosen_move = moves[torch.argmax(scores)]
>>>>>>> 743b195641a46de1a6d5c249c2b1e5c0bfe425e8
        return chosen_move

    def play(self, board, color):
        board.push(self.get_next_move(board, color))


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
<<<<<<< HEAD
=======
    print(chess.pgn.Game().from_board(board))

>>>>>>> 743b195641a46de1a6d5c249c2b1e5c0bfe425e8
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
    stock = StockfishEngine(time=2, depth=1)
    stock2 = StockfishEngine(time=2, depth=11)
    cnn = CNN_Net()
    cnn.load_state_dict(torch.load("../saves/cnn/cnnfinal.pt"))
    cnn.eval()
    cnn_engine = CNN_Engine(cnn.to(device), device)
    compare_engines(100,stock, cnn_engine, True)
    stock.quit()
    stock2.quit()
    random_engine.quit()
