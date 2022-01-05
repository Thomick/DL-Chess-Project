# Data generator

import chess
import chess.pgn
import chess.engine
import io
import numpy as np
import sys

mate_score = 50000

elo_limit = 2750

engine_path = "/home/gabrielj/myapps/stockfish_14.1_linux_x64/stockfish_14.1_linux_x64"

# penser à ajouter un threshold d'elo

def extract_games_from_file(path):
    games_file = open(path)
    games_string = games_file.readlines()
    list_games_string = []
    game = []
    j = 0
    for i in games_string:
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
    for x in list_games_string:
        g = chess.pgn.read_game(io.StringIO(x))
        #print(x)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #print(g)
        #print("######################################")
        if g.headers["Event"]=="?" :
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

    encodings = []
    scores = []

    #do while
    while True:
        encodings.append(encode_board(game.board()))

        info = engine.analyse(game.board(), chess.engine.Limit(depth=20))

        score = info["score"].white().score(mate_score=mate_score)
        scores.append(score)
        
        #print(game.board())
        #print("Score = ", score)

        if game.is_end() :
            break

        game = game.next()

    engine.quit()

    return np.array(encodings), scores
    

id = 315

caruana_games = extract_games_from_file("../data/raw_data/Caruana.pgn")

print("There are games : ", len(caruana_games))

codings, scores = extract_all_positions_encodings(caruana_games[id].game())

print(caruana_games[id])
print(codings.shape)
#print(codings[codings.shape[0]-1])
print(scores)

