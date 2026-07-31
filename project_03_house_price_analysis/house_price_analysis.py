import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class HousePriceAnalysis:
    def __init__(self,file_path) -> None:
        self.file_path = Path(file_path)
        self.output_dir = Path('outputs')
        self.df = None
        sns.set_theme(style="whitegrid")

    def load_data(self) -> None:
        try:
            self.df = pd.read_csv(self.file_path,header=None)
            print("***House Price Dataset***")
            print("Dataset Loaded Successfully")
            print(f"File : {self.file_path}")
        except FileNotFoundError:
            print(f"Dataset Not Found: {self.file_path}")
            raise
        except pd.errors.EmptyDataError:
            print(f"The CSV File is Empty:")
            raise
    def dataset_information(self) -> None:
        if self.df is None:
            raise ValueError("Dataset has not been Loaded.")
        print("*****Dataset Information*****")
        print("\nFirst 5 Rows")
        print(self.df.head)
        print("\nDataset Shape:",self.df.shape)
        column = self.df[1]
        print("\nColumn Names:{column}")
        print("Data Type:",self.df.dtypes)
        print("\nPandas Info")
        self.df.info()
        print("\nMissing Values --> ",self.df.isnull().sum())
        print("\nDuplicates Rows --> ",self.df.duplicated().sum())




    def run(self):
        self.load_data()
        self.dataset_information()

if __name__ == "__main__":
    analysis = HousePriceAnalysis("datasets/house_prices_prediction.csv")
    analysis.run()
