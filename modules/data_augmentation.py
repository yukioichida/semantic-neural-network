import nltk
from nltk.corpus import wordnet as wn

# wordnet tags:
# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

a = "https://api.explosion.ai/sense2vec/find?word=very&sense=ADV"

def get_syn(word, pos):
    try:
        # map part of speech tags to wordnet
        # Usando sinonimos de adjetivos e adverbios mostrou bom resultado
        pos = {'JJ': wn.ADJ, 'RB': wn.ADV}[pos]
    except LookupError:
        # print "Unexpected error:", sys.exc_info()[0]
        # or just return the original word
        # print("OUCH {} {}".format(word, pos))
        print("POS {} ignored".format(pos))
        return []

    synsets = wn.synsets(word, pos)
    synonyms = []
    print word
    if synsets:
        for synset in synsets:
            print "Synset: {}, lemmas: {}, similary tos: {}".format(synset, synset.lemma_names(), synset.similar_tos())
            for lemma in synset.lemma_names():
                if lemma != word:
                    synonyms.append(lemma)
    print synonyms
    return synonyms or []

    # print("Word {}".format(word))
    #for synset in synsets:
    #   print "Synset: {}".format(synset)
    #    for lemma in synset.lemma_names():
    #        print lemma
    #    print "============================"



#sentence = "A big wall bewteen us is beating frequently"
sentence = "he is smart, very good"
words = nltk.word_tokenize(sentence)
pos_tag = nltk.pos_tag(words)
print pos_tag

new_sentence = []
synonyms = {}
min_syn = 9999
changes = 0
for (word, pos) in pos_tag:
    # first two letters of postag
    print pos[:2]
    synonyms[word] = get_syn(word, pos[:2])
    if len(synonyms) < min_syn:
        min_syn = len(synonyms)


for i in range(0, min_syn):
    new_sentence = []
    for word in words:
        if synonyms[word]:
            new_sentence.append(synonyms[word][i])
        else:
            new_sentence.append(word)
    print new_sentence


print wn.synsets('smart', wn.ADJ)
smart = wn.synset('smart.a.01')
print smart
intelligent = wn.synset('intelligent.a.01')
print intelligent
print wn.lch_similarity(smart, intelligent)
#data_augment(sentence, pos_tag)