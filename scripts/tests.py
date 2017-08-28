'''
building the vocabulary

from keras.preprocessing.text import Tokenizer

text = ["focus on the work keras", "Test in the keras"]
text2 = ["ths is a second collection", "I need to write english soon as possible keras"]

tokenizer = Tokenizer(100)
tokenizer.fit_on_texts(text)
tokenizer.fit_on_texts(text2)

#print(tokenizer.num_words)
print(tokenizer.texts_to_sequences(text2))
print(tokenizer.texts_to_sequences(text))
print('Found %s unique tokens.' % len(tokenizer.word_index))
'''

'''
split test and train sets


import numpy as np
from sklearn.model_selection import train_test_split

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
y = ['a', 'b', 'c', 'd']

Xtrain, Xtest, X2train, X2test, Ytrain, Ytest = train_test_split(np.array(a), np.array(b), np.array(y), shuffle = False)

print(Xtrain)
print(Xtest)
print(X2train)
print(X2test)
print(Ytrain)
print(Ytest)
'''

a = '1.250'
print(float(a) - 1)