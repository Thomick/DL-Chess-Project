import chess
import math
import random
import sys
import copy
from threading import Thread
import numpy as np
from torch.multiprocessing import Pool, Process, set_start_method


def alphaBetaMax(board,alpha, beta, depth_left,eval_f,color):
    if depth_left == 0:
        if board.is_checkmate():
            return 50000
        return  eval_f(board)*color

    possibleMoves = board.legal_moves

    for m in possibleMoves:

        board.push(m)
        score = alphaBetaMin(board,alpha,beta, depth_left-1, eval_f, color)
        board.pop()

        if score >= beta :
            return beta

        if score > alpha:
            alpha = score

    return alpha
    
def alphaBetaMin(board,alpha, beta, depth_left,eval_f,color):
    if depth_left == 0:
        if board.is_checkmate():
            return -50000
        return  eval_f(board)*color

    possibleMoves = board.legal_moves
    for m in possibleMoves:

        board.push(m)
        score = alphaBetaMax(board, alpha,beta, depth_left-1, eval_f, color)
        board.pop()

        if score <= alpha :
            return alpha
        if score < beta:
            beta = score

    return beta

def alphaBetaRoot(board, eval_f, color, depth):
    possibleMoves = board.legal_moves
    bestMove = -99999
    bestMoveFinal = None
    for move in possibleMoves:
        board.push(move)
        value = alphaBetaMax(board, -100000,100000, depth-1, eval_f, color)
        board.pop()
        if( value > bestMove):
            #print("Best score: " ,str(bestMove))
            #print("Best move: ",str(bestMoveFinal))
            bestMove = value
            bestMoveFinal = move
    return bestMoveFinal

def wrapper_alphaBetaMax(results,i,board,alpha, beta, depth_left,eval_f,color):
    results[i] = alphaBetaMax(board,alpha, beta, depth_left,eval_f,color)

def alphaBetaRoot_mt(board, eval_f, color, depth):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    possibleMoves = board.legal_moves

    args = []
    for move in possibleMoves:
        cboard = copy.deepcopy(board)
        cboard.push(move)
        args += [(cboard, -100000, 100000, depth-1, eval_f, color)]

    with Pool(10) as pool:
        results = pool.starmap(alphaBetaMax, args)

    best_move = list(possibleMoves)[np.argmax(results)]

    return best_move
