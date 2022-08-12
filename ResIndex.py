from dataclasses import dataclass

@dataclass
class ResIndex:
    df_name = ""
    fold = 0
    iter = 0

    # parameterized constructor
    def __init__(self,i, j, k):
        self.df_name = i
        self.fold = j
        self.iter = k

    def get_df(self):
        return self.df_name

    def get_fold(self):
        return self.fold

    def get_iter(self):
        return self.iter

