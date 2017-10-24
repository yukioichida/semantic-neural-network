# -*- coding: utf-8 -*-
'''
import nltk
import sys
from nltk.corpus import wordnet as wn

import random
import pickle

#a = pickle.load(open('synsem_conv.p', 'rb'))
#print a


for ss in wn.synsets('hard', wn.ADJ):
    print ss, ss.hypernyms()
    print ss.similar_tos()

# wup_similarity = Wu-Palmer Similarity
print wn.synset('sword.n.01').wup_similarity(wn.synset('weapon.n.01'))


def synonimize(word, pos=None):
    """ Get synonyms of the word / lemma """
    try:
        # map part of speech tags to wordnet
        # Usando sinonimos de adjetivos e adverbios mostrou bom resultado
        pos = {'NN': wn.NOUN, 'JJ': wn.ADJ, 'VB': wn.VERB, 'RB': wn.ADV}[pos[:2]]
    except LookupError:
        #print "Unexpected error:", sys.exc_info()[0]
        # or just return the original word
        #print("OUCH {} {}".format(word, pos))
        return [word]

    synsets = wn.synsets(word, pos)
    synonyms = []
    for synset in synsets:
        for sim in synset.similar_tos():
            synonyms += sim.lemma_names()

    # return list of synonyms or just the original word
    return synonyms or [word]


def vary_sentence(sentence):
    """ Create variations of a sentence using synonyms """
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    words = []
    for (word, pos) in pos_tags:
        synonyms = synonimize(word, pos)
        picked = random.choice(synonyms)
        words.append(picked)

    return " ".join(words)


if __name__ == "__main__":
    for i in range(0, 10):
        print(vary_sentence("this hard work is very awesome"))


'''

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(['Eu estudo computação', 'Eu estudo aprendizado automático'])
print(tokenizer.word_index)

