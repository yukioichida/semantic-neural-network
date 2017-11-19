# -*- coding: utf-8 -*-
import pandas as pd
import xml.etree.ElementTree as ElementTree


class Dataset:
    def data_frame(self):
        return self.df


class STSDataset(Dataset):
    def __init__(self, filename):
        self.df = pd.read_csv(filename, sep='\t', encoding="utf-8", names=['label', 's1', 's2'], quoting=3)

    def name(self):
        return "sts"


class MSRPDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename, encoding="utf-8", sep='\t', quoting=3)
        self.df = df.rename(columns={'#1 String': 's1', '#2 String': 's2', 'Quality': 'label'})

    def name(self):
        return "MSRP"


class SICKDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t', encoding="utf-8")
        df = df[df.entailment_judgment != 'CONTRADICTION']
        self.df = df.rename(columns={'sentence_A': 's1', 'sentence_B': 's2', 'relatedness_score': 'label'})

    def name(self):
        return "sick_2014"


class SICKFullDataset(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename, sep='\t', encoding="utf-8", quoting=3)
        self.df = df.rename(columns={'sentence_A': 's1', 'sentence_B': 's2', 'relatedness_score': 'label'})

    def name(self):
        return "full_sick_2014"


class AssinDataset(Dataset):
    def __init__(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        similarities = []
        s1 = []
        s2 = []
        for child in root:
            similarities.append(child.attrib['similarity'])
            s1.append(child.find('t').text)
            s2.append(child.find('h').text)
        data = {'s1': s1, 's2': s2, 'label': similarities}
        self.df = pd.DataFrame(data=data)
