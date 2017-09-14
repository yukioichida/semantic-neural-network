from modules.input_data import ProcessInputData
from modules.datasets import STSDataset
import pandas as pd

process = ProcessInputData()
file = 'resource_test.tsv'
df_test = STSDataset(file).data_frame()

s1, s2, labels = process.pre_process_data(df_test)

for i in range(0, len(s1)):
    print("%s - %s - %s" % (s1[i], s2[i], labels[i]))

