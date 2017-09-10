
import os
from pathlib import Path

dir = os.path.dirname(__file__)

print (Path(dir).parent)
#print(stats.pearsonr([1, 2, 3], [0.8, 1.9, 3.2]))

import time
milli_sec = int(round(time.time() * 1000))
print(milli_sec)


a = 3.7889
b = (a - 1) / 4

print (b)



b = a / 5
print(b)

import scipy.stats as stats
a = [1.2, 3.4, 4.6]
b = [2.0, 2.2, 3.0]

print(stats.pearsonr(a, b))
