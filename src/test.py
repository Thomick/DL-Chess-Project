import io
import sys
import os
import re

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def extract_games_from_file(path):
    blank_line_regex = r"(?:\r?\n){2,}"
    games_file = open(path)
    myarray = re.split(blank_line_regex, games_file.read())
    for p1,p2 in pairwise(myarray):
        print(p1+"\n\n"+p2)
        print("Next Line")

extract_games_from_file("../data/raw_data/classicald4/Colle.pgn")
