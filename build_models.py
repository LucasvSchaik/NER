"""
FILE: features.py
Author: Jip Heimeriks, Milo Broerse, Lucas van Schaik
"""
from custom_chunker import ConsecutiveNPChunker
from nltk.corpus import conll2002 as conll

import pickle
import inspect
import features

training = conll.chunked_sents("ned.train")
# Get all members of the module
members = inspect.getmembers(features)
all_features = [member[1] for member in members if inspect.isfunction(member[1])]

for feature in all_features:
        name = feature.__name__
        model = ConsecutiveNPChunker(feature, training)
        if name == "features3":
                output = open("best.pickle", "wb")
                pickle.dump(model, output)
                output.close()
        if name == "features4":
                output = open("other.pickle", "wb")
                pickle.dump(model, output)
                output.close()
        else:
                output = open(f"{name}.pickle", "wb")
                pickle.dump(model, output)
                output.close()
