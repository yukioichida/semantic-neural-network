import pandas as pd

class Dataset:

    def data_frame(self):
        return self.df


class STSDataset(Dataset):

    def __init__(self, filename):
        self.df = pd.read_csv(filename, sep='\t', encoding="utf-8", names=['label', 's1', 's2'])


class QuoraQuestionsDataset(Dataset):

    def __init__(self, filename):
        df = pd.read_csv(filename, encoding="utf-8")
        self.df = df.rename(columns={'question1': 's1', 'question2': 's2', 'is_duplicate': 'label'})


class SICKDataset(Dataset):

    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t', encoding="utf-8")
        df = df[df.entailment_judgment != 'CONTRADICTION']
        self.df =  df.rename(columns={'sentence_A': 's1', 'sentence_B': 's2', 'relatedness_score': 'label'})
