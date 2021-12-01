import pickle as pkl
import numpy as np
from sklearn.utils.extmath import cartesian

LETTERS = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h",
9: "i", 11: "k", 12: "l", 13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r",
19: "s", 20: "t", 21: "u", 22: "v", 23: "w", 24: "x", 25: "y"}

def process_words(filepath):
    word_dict = {}

    with open(filepath) as f:
        words = f.readlines()

    for word in words:
        word = word.strip()
        key = len(word)
        if 'j' not in word and 'z' not in word:
            if key in word_dict:
                word_dict[key].add(word)
            else:
                word_dict[key] = {word}
    
    pkl.dump(word_dict, open("../data/words.p", "wb"))

def convert_to_letter(x):
    return LETTERS[x]

def find_word(l, unpickled=False):
    """
    l is a np.ndarray, where each row represents the top N guesses
    for each sign language gesture.
    """
    if not unpickled:
        words = pkl.load(open("../data/words.p", "rb"))
    
    f = np.vectorize(convert_to_letter)
    l = f(l)

    len_words = words[l.shape[0]]
    for try_word in cartesian(l):
        try_word = ''.join(try_word)
        if try_word in len_words:
            return try_word
    
    return "Error: no words found"