# Data generator

import chess.pgn
import io
import numpy as np


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

    list_games = [ chess.pgn.read_game(io.StringIO(g)) for g in list_games_string]  

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
            [board.is_checkmate()],\ # je suis plutôt incertain de la nécessité de ces deux dernières informations, par forcément nécessaire dans l'encoding
            [board.is_stalemate()],\
            ), axis=0)
    return encoding


def extract_all_positions_encodings(game):
    encodings = []

    #do while
    while True:
        encodings.append(encode_board(game.next().board()))

        game = game.next()

        if game.is_end() :
            break

    return np.array(encodings)
    

id = 2500

caruana_games = extract_games_from_file("../data/Caruana.pgn")

codings = extract_all_positions_encodings(caruana_games[id].game())

print(caruana_games[id])
print(codings.shape)
print(codings[codings.shape[0]-1])


