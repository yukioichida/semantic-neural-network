import pandas as pd
from pandas import DataFrame

class Dataset:

    def data_frame(self) -> DataFrame:
        return self.df


class STSDataset(Dataset):

    def __init__(self, filename):
        self.df = pd.read_csv(filename, sep='\t', encoding="utf-8", names=['label', 's1', 's2'])

    def name(self):
        return "sts"


class QuoraQuestionsDataset(Dataset):

    def __init__(self, filename):
        df = pd.read_csv(filename, encoding="utf-8")
        self.df = df.rename(columns={'question1': 's1', 'question2': 's2', 'is_duplicate': 'label'})

    def name(self):
        return "quora-kaggle"


class SICKDataset(Dataset):

    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t', encoding="utf-8")
        df = df[df.entailment_judgment != 'CONTRADICTION']
        self.df =  df.rename(columns={'sentence_A': 's1', 'sentence_B': 's2', 'relatedness_score': 'label'})

    def name(self):
        return "sick_2014"


class SICKFullDataset(Dataset):

    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t', encoding="utf-8")
        self.df =  df.rename(columns={'sentence_A': 's1', 'sentence_B': 's2', 'relatedness_score': 'label'})

    def name(self):
        return "full_sick_2014"