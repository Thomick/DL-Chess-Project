# Data generator

import chess
import chess.pgn
import chess.engine
import io
import numpy as np
import sys
import os
import torch
from torch.utils.data import Dataset
from progressbar import progressbar

stream = os.popen("which stockfish")

engine_path = stream.read().split('\n')[0]

def extract_games_from_file(path):
    games_file = open(path)
    games_string = games_file.readlines()
    list_games_string = []
    game = []
    j = 0
    print("*Parsing file*")
    for i in progressbar(games_string):
        if i == "\n":
            j += 1
        if j == 2:
            j = 0
            game_str = ''.join(game)
            list_games_string.append(game_str)
            game = []
        game.append(i)
    if game :
        game_str = ''.join(game)
        list_games_string.append(game_str)

    list_games = [] # [ chess.pgn.read_game(io.StringIO(g)) for g in list_games_string]  
    print("*Elo filtering*")
    for x in progressbar(list_games_string):
        g = chess.pgn.read_game(io.StringIO(x))

        if g.headers["Event"]=="?" :
            print(x)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(g)
            print("######################################")

            sys.exit()

        #print("WhiteElo : ",g.headers["WhiteElo"], "BlackElo:",g.headers["BlackElo"])
        try:
            if int(g.headers["WhiteElo"]) >= elo_limit and int(g.headers["BlackElo"]) >= elo_limit:
                list_games.append(g)
        except:
            pass

    return list_games


# a voir si on peut ajouter dans l'encoding le nombre de half moves depuis la dernière prise/pawn move, pour émuler la règle des 50 coups
# nous pouvons faire de même pour un matériel insuffisant que force la draw
#pourquoi pas ajouter le nombre de pièces attaquées et non défendues
def encode_board(board):
    encoding = np.concatenate(( \
            board.pieces(chess.PAWN, chess.WHITE).tolist(),\
            board.pieces(chess.BISHOP, chess.WHITE).tolist(),\
            board.pieces(chess.KNIGHT, chess.WHITE).tolist(),\
            board.pieces(chess.ROOK, chess.WHITE).tolist(),\
            board.pieces(chess.QUEEN, chess.WHITE).tolist(),\
            board.pieces(chess.KING, chess.WHITE).tolist(),\
            board.pieces(chess.PAWN, chess.BLACK).tolist(),\
            board.pieces(chess.BISHOP, chess.BLACK).tolist(),\
            board.pieces(chess.KNIGHT, chess.BLACK).tolist(),\
            board.pieces(chess.ROOK, chess.BLACK).tolist(),\
            board.pieces(chess.QUEEN, chess.BLACK).tolist(),\
            board.pieces(chess.KING, chess.BLACK).tolist(),\
            [board.has_kingside_castling_rights(chess.WHITE)],\
            [board.has_queenside_castling_rights(chess.WHITE)],\
            [board.has_kingside_castling_rights(chess.BLACK)],\
            [board.has_queenside_castling_rights(chess.BLACK)],\
            [board.turn],\
            [board.is_check()],\
            ), axis=0)
    return encoding

def extract_all_positions_encodings(game):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    encodings = [ [encode_board(game.board()),\
            engine.analyse(game.board(), chess.engine.Limit(depth=engine_depth))["score"].white().score(mate_score=mate_score)\
            ] for g in game.mainline()]

    engine.quit()

    return np.array(encodings, dtype = object)



class GMChessDataset(Dataset):
    """Grandmaster Chess games dataset"""

    def __init__(self, encodings):
        """
        Args:
            encodings (np.array): Array of shape (n,2) containings
            positions encodings and stockfish evaluations
        """
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        sample = {'position': self.encodings[idx][0], 'evaluation': self.encodings[idx][0]}
        return sample
    

if __name__ == "__main__":

    mate_score = 50000

    elo_limit = 2750


    engine_depth = 1

    data_files = [
        "../data/raw_data/Modern.pgn"
            ]

    data = np.empty((0,2))

    for p in data_files:
        print("Handling file : ", p)
        games = extract_games_from_file(p)
        print("Games found : ", len(games))
        print("*Extracting positions from games*")
        for g in progressbar(games):
            encodings = extract_all_positions_encodings(g.game())
            data = np.concatenate((data,encodings))
        print("\n\n")

    print("Positions loaded =", data.shape)

    GM_set = GMChessDataset(data)

    torch.save(GM_set, "../data/GM_set.pt")
