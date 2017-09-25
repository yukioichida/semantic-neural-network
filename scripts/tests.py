
import os
from pathlib import Path

dir = os.path.dirname(__file__)

print (Path(dir).parent)
#print(stats.pearsonr([1, 2, 3], [0.8, 1.9, 3.2]))

import time
milli_sec = int(round(time.time() * 1000))
print(milli_sec)

import numpy as np
v = [1, 0.6, 2, 4.7, 0.7]
a = np.array(v)

print(np.clip(a, 1, 5))

from keras.preprocessing.text import Tokenizer

t = Tokenizer()
t.fit_on_texts(['eu estou cansado'])
print(t.word_index)
t.fit_on_texts(['estou com esperanças'])
print(t.word_index)
