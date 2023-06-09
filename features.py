"""
FILE: features.py
Author: Jip Heimeriks, Milo Broerse, Lucas van Schaik
"""
import csv
from operator import itemgetter

name_list =[]
with open("./VNC2013.csv", 'r', encoding='utf-8') as file:
    csvreader = csv.reader(file)
    name_list = [(row[0], row[2]) for row in csvreader]
    name_list.sort(key=itemgetter(1), reverse=True)

common_name_list = name_list[:200]


def test_features(sentence, i, history):
    """dummy Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """
    word, pos = sentence[i]
    return {
        "pos": pos,
        "whole history": tuple(history)
            }


def features1(sentence, i, history):
    """features:
            -pos
            -word
            -previous pos
            -previous word
            -word begins with capital
            -previous word begins with capital
            -lenght of word
            -word contains numbers
            -position of word in sentence
            -whole history
    """
    word, pos = sentence[i]
    Capital = False
    if word[0].isupper():
        Capital = True
    pre_word, pre_pos = sentence[i-1]
    pre_capital = False
    if pre_word[0].isupper():
        pre_capital = True
    return{
        "pos": pos,
        "word": word,
        "pre pos": pre_pos,
        "pre word": pre_word,
        "capital": Capital,
        "pre capital": pre_capital,
        "lenght": len(word),
        "numbers in word":any(char.isdigit() for char in word),
        "position": i,
        "whole history": tuple(history)
    }


def features2(sentence, i, history):
    """ features:
            -pos
            -previous word
            -previous pos
            -word is title
            -whole history
    """
    word, pos = sentence[i]

    pre_word, pre_pos = sentence[i-1]
    return {
        "pos": pos,
        "pre word":pre_word,
        "pre pos":pre_pos,
        "title": word.istitle(),
        "whole history": tuple(history)
            }


def features3(sentence, i, history):
    """features:
            -pos
            -word
            -word begins with capital
            -previous word begins with capital
            -previous pos
            -previous word
            -word contains numbers
            -whole history
    """
    word, pos = sentence[i]

    pre_word, pre_pos = sentence[i-1]
    Capital = False
    if word[0].isupper():
        Capital = True
    pre_capital = False
    if pre_word[0].isupper():
        pre_capital = True
    return {
        "pos": pos,
        "word": word,
        "Capital": Capital,
        "pre Capital": pre_capital,
        "pre word": pre_word,
        "pre pos": pre_pos,
        "numbers":any(char.isdigit() for char in word),
        "whole history": tuple(history)
    }


def features4(sentence, i, history):
    """features:
            -pos
            -word
            -lenght of word
            -word begins with capital
            -word contains numbers
            -previous word
            -previous pos
            -word is title
            -first two letters of word
            -whole history
    """
    word, pos = sentence[i]
    Name = False
    if word in common_name_list:
        Name = True
    pre_word, pre_pos = sentence[i-1]
    Capital = False
    if word[0].isupper():
        Capital = True
    return {
        "pos": pos,
        "word": word,
        "lenght": len(word),
        "capital": Capital,
        "numbers":any(char.isdigit() for char in word),
        "pre word":pre_word,
        "pre pos":pre_pos,
        "title": word.istitle(),
        "first two letters": word[:1],
        "whole history": tuple(history)
            }
