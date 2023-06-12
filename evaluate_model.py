"""
FILE: features.py
Author: Jip Heimeriks, Milo Broerse, Lucas van Schaik
"""
from custom_chunker import ConsecutiveNPChunker
from nltk.corpus import conll2002 as conll
import pickle
import features
import build_models

with open("./evaluation_output.txt", "w") as eva:
    eva.write("output from evaluation of classifiers. \n\n")

test_sents = conll.chunked_sents("ned.testa")

for classifier in build_models.all_features:
    name = classifier.__name__
    if name == "features3":
        input = open("best.pickle", "rb")
        Classifier = pickle.load(input)
        with open("./evaluation_output.txt", "a") as eva:
            eva.write(f"{name} pickled as best.pickle. \n\n",name,"\n",name.__doc__ ,
                      "\n\n",Classifier.accuracy(test_sents))
        input.close()
    if name == "features4":
        input = open("other.pickle", "rb")
        Classifier = pickle.load(input)
        with open("./evaluation_output.txt", "a") as eva:
            eva.write(f"{name} pickled as other.pickle","\n\n",name,"\n",name.__doc__ ,
                      "\n\n",Classifier.accuracy(test_sents))
        input.close()
    else:
        input = open(f"{name}.pickle", "rb")
        Classifier = pickle.load(input)
        with open("./evaluation_output.txt", "a") as eva:
            eva.write(f"{name} pickled as {name}.pickle","\n\n",name,"\n",name.__doc__ ,
                      "\n\n",Classifier.accuracy(test_sents))
        input.close()
