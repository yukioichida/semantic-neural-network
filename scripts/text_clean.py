import os
import re

DATASET_BASEDIR = "..\\..\\datasets\\"
# Paraphrase dataset
UNIFIED_PP_DATASET = os.path.join(DATASET_BASEDIR, "pp\pp-unified.tsv")
UNIFIED_PP_DATASET_OUTPUT = os.path.join(DATASET_BASEDIR, "pp\pp-unified-processed.tsv")
# STS english dataset
UNIFIED_STS_DATASET = os.path.join(DATASET_BASEDIR, "similarity\en\sts.tsv")
UNIFIED_STS_DATASET_OUTPUT = os.path.join(DATASET_BASEDIR, "similarity\en\sts-processed.tsv")

def normalize_acronyms(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    return text

def normalize_symbols(text):
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def normalize(text):
    text = normalize_acronyms(text)
    text = normalize_symbols(text)
    return text

def normalize_line(line):
    tokens = line.split('\t')
    sent_1 = normalize(tokens[1])
    sent_2 = normalize(tokens[2])
    result_line = "%s\t%s\t%s" % (tokens[0], sent_1, sent_2)
    return result_line

#line = """4.000	University of Michigan President Mary Sue Coleman said in a statement on the university's Web site, "Our fundamental values haven't changed.	"Our fundamental values haven't changed," Mary Sue Coleman, president of the university, said in a statement in Ann Arbor."""
#print(normalize_line(line))

with open(UNIFIED_STS_DATASET, 'r', encoding='utf-8') as input_file:
    with open(UNIFIED_STS_DATASET_OUTPUT, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            output_file.write(normalize_line(line) + '\n')

with open(UNIFIED_PP_DATASET, 'r', encoding='utf-8') as input_file:
    with open(UNIFIED_PP_DATASET_OUTPUT, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            output_file.write(normalize_line(line) + '\n')
