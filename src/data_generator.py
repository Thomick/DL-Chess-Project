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
import re
from collections import OrderedDict
from os.path import isfile, join

stream = os.popen("which stockfish")

engine_path = stream.read().split('\n')[0]

engine_depth = 15
mate_score = 50000


# -------------------------------------------------------------------------------
# This class defines colors for the terminal output
# -------------------------------------------------------------------------------
class bcolors:
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# -------------------------------------------------------------------------------
# This function is used to iterate two by two over an iterable
# -------------------------------------------------------------------------------
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

# -------------------------------------------------------------------------------
# This function is used to sort games by elo
# -------------------------------------------------------------------------------


def elo_sort(g):
    try:
        return int(g.headers["WhiteElo"]) + int(g.headers["BlackElo"])

    except:
        return 0


# -------------------------------------------------------------------------------
# This function loads games from a pgn file.
# It will load a maxmimum number of games, set by n_threshold
n_threshold = 50
# We take the best games
# -------------------------------------------------------------------------------


def extract_games_from_file(path, log_f):
    games_file = open(path, encoding="latin-1")

    print("* Parsing *")
    blank_line_regex = r"(?:\r?\n){2,}"
    str_data = re.split(blank_line_regex, games_file.read())
    list_games = []

    for p1, p2 in progressbar(pairwise(str_data), redirect_stdout=True):
        pgn = p1+"\n\n"+p2

        g = chess.pgn.read_game(io.StringIO(pgn))

        if g.headers["Event"] == "?":
            log_f.write("[DEBUG] PARSING ERROR IN {}\n".format(path))
            log_f.write(pgn)
            log_f.write("\n------------------------------------------\n")
            print(g, file=log_f)
            log_f.write("\n")
            log_f.write("[END DEBUG] PARSING ERROR\n")
            print(bcolors.FAIL + bcolors.BOLD + "[FATAL PARSING ERROR]" + bcolors.ENDC + " Check log to see details")

            break

        list_games.append(g)

    list_games.sort(reverse=True, key=elo_sort)

    number_of_games = len(list_games)

    to_take = min(n_threshold, number_of_games)

    return list_games[0:to_take], to_take, number_of_games

# -------------------------------------------------------------------------------
# This function is used to encode a board into a vector
# -------------------------------------------------------------------------------


def encode_board(board):
    encoding = np.concatenate((
        board.pieces(chess.PAWN, chess.WHITE).tolist(),
        board.pieces(chess.BISHOP, chess.WHITE).tolist(),
        board.pieces(chess.KNIGHT, chess.WHITE).tolist(),
        board.pieces(chess.ROOK, chess.WHITE).tolist(),
        board.pieces(chess.QUEEN, chess.WHITE).tolist(),
        board.pieces(chess.KING, chess.WHITE).tolist(),
        board.pieces(chess.PAWN, chess.BLACK).tolist(),
        board.pieces(chess.BISHOP, chess.BLACK).tolist(),
        board.pieces(chess.KNIGHT, chess.BLACK).tolist(),
        board.pieces(chess.ROOK, chess.BLACK).tolist(),
        board.pieces(chess.QUEEN, chess.BLACK).tolist(),
        board.pieces(chess.KING, chess.BLACK).tolist(),
        [board.has_kingside_castling_rights(chess.WHITE)],
        [board.has_queenside_castling_rights(chess.WHITE)],
        [board.has_kingside_castling_rights(chess.BLACK)],
        [board.has_queenside_castling_rights(chess.BLACK)],
        [board.turn],
        [board.is_check()],
    ), axis=0)
    encoding = encoding*1.
    return encoding

# -------------------------------------------------------------------------------
# This function is used to extract all positions encodings from a game
# -------------------------------------------------------------------------------


def extract_all_positions_encodings(game):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    encodings = [[encode_board(g.board()),
                  engine.analyse(g.board(), chess.engine.Limit(depth=engine_depth))[
        "score"].white().score(mate_score=mate_score)/100.0
    ] for g in game.mainline()]

    engine.quit()

    return np.array(encodings, dtype=object)

# -------------------------------------------------------------------------------
# This function is used to hash an encoding
# -------------------------------------------------------------------------------


def hash_encoding(encoding):
    return encoding.tobytes()


# -------------------------------------------------------------------------------
# This class defines a dataset of positions encodings
# -------------------------------------------------------------------------------
class ChessDataset(Dataset):
    """Grandmaster Chess games dataset"""

    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (string): path to directory containing pgn files
        """
        self.encodings = OrderedDict()

        # for logging
        log = open("../log/" + os.path.basename(dataset_dir) + ".log", "w")

        print("[DATASET] Loading {}".format(dataset_dir))
        log.write("[DATASET] Loading {}\n".format(dataset_dir))

        # name
        self.name = os.path.basename(dataset_dir)

        # Iterating over all files in the directory
        total_positions = 0
        files = [join(dataset_dir, f) for f in os.listdir(dataset_dir) if isfile(join(dataset_dir, f))]
        for file in progressbar(files, redirect_stdout=True):
            n_positions = 0
            print("[HANDLING FILE] : {}".format(file))
            log.write("[HANDLING FILE] : {}\n".format(file))
            games, n_loaded, n_total = extract_games_from_file(file, log)
            print("[REPORT] Loaded {} games out of {}".format(n_loaded, n_total))
            log.write("[REPORT] Loaded {} games out of {}\n".format(n_loaded, n_total))
            for g in progressbar(games, redirect_stdout=True):
                encodings = extract_all_positions_encodings(g)
                n_positions += encodings.shape[0]
                for e in encodings:
                    self.encodings[hash_encoding(e)] = e
            print("[LOADED] Loaded {} positions from {}".format(n_positions, file))
            log.write("[LOADED] Loaded {} positions from {}\n".format(n_positions, file))

            total_positions += n_positions

        print("[LOADED] Loaded a total of {} positions".format(total_positions))
        log.write("[TOTAL LOADED] Loaded a total of {} positions\n".format(total_positions))

        print("[IN HASHMAP] {} positions were added to the hashmap of encodings".format(len(self.encodings)))
        log.write("[IN HASHMAP] {} positions were added to the hashmap of encodings\n".format(len(self.encodings)))

        log.close()

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        sample = (self.encodings.items()[idx][0], self.encodings.items()[idx][1])
        return sample

    def save(self):
        torch.save(self, "../data/"+self.name+".pt")

    def merge(self, Other):
        self.encodings.update(Other.encodings)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide datasets")
        sys.exit()

    datasets = sys.argv[1:]

    for dataset in progressbar(datasets, redirect_stdout=True):
        set = ChessDataset(dataset)
        set.save()
