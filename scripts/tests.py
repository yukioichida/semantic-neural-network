
import os
from pathlib import Path

dir = os.path.dirname(__file__)

print (Path(dir).parent)
#print(stats.pearsonr([1, 2, 3], [0.8, 1.9, 3.2]))

import time
milli_sec = int(round(time.time() * 1000))
print(milli_sec)







