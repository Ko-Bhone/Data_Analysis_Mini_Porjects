from pathlib import Path
import numpy as np
import pandas as pd

class HousePriceAnalysis:
    def __init__(self,file_path):
        self.file_path = Path(file_path)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok = True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path, index_col = 0)
            print("******Dataset Loaded Successfully******")
            print(f"File:{self.file_path}")
        except FileNotFoundError:
            print(f"Dataset Not Found:{self.file_path}")
            raise
        except pd.errors.EmptyDataError:
            print(f"The CSV File is Empty:{self.file_path}")
            raise

    def dataset_info(self) -> None:
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        print(f"\n===Dataset Info===")
        print(f"\nFirst Five Rows...")
        print(self.df.head())
        print(f"\nDataset Shape....")
        rows,cols = self.df.shape
        print(f"Rows:{rows}")
        print(f"Columns:{cols}")
        print(f"\nColumn Name...")
        print(self.df.columns)
        print(f"\nData Type...")
        print(self.df.dtypes)
        print(f"\nPandas Info...")
        print(self.df.info())
        print(f"\nMissing Values...")
        print(self.df.isnull().sum())
        print(f"\nDuplicate Values...")
        print(self.df.duplicated().sum())

    def clean_data(self)-> None:
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        print(f"\n===DATASET CLEANING===")
        print(f"Remove Duplicate Rows")
        duplicate_count = (self.df.duplicated().sum())
        if duplicate_count > 0:
            self.df.drop_duplicates(inplace = True)
            print(f"Removed Duplicate Rows:{duplicate_count}")
        else:
            ("No Duplicate Rows Found")
        #Handle missing numerical values
        numerical_columns = ["Area","Bedrooms","Bathrooms","Stories","Parking","Age","Price"]
        for column in numerical_columns:
            if self.df[column].isnull().any():
                median_value = (self.df[column].fillna(median_value))
                print(f"Filled Missing Values:{column}")
        #Handle Missing Categorical values
        categorical_columns = ["Location","Property_Type"]
        for column in categorical_columns:
            if self.df[column].isnull().any():
                mode_value = (self.df[column].mode()[0])
                print(f"Filled Missing Values:{column}")
        #Data Type Conversion
        integer_columns = ["House_ID","Area","Bedrooms","Bathrooms","Stories","Parking","Age"]
        for column in integer_columns:
            self.df[column] = (self.df[column].astype(int))
            self.df["Price"] = (self.df["Price"].astype(float))
        #Validate Price
        invalid_prices = (self.df["Price"] <=0 ).sum()
        if invalid_prices > 0:
            self.df = self.df[self.df["Price"] > 0]
            print(f"Removed Invalid Prices:{invalid_prices}")
        #Final Missing Value Check
        print(f"\nRemaining Missing Values...")
        print(self.df.isnull().sum())
        print("\nData Cleaning Completed")

    #Basic Statistic
    def basic_statistics(self) -> None:
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        print(f"===PRICE STATISTICS===")
        price = self.df["Price"].to_numpy()
        price_mean = np.mean(price)
        print(f"Mean Price: {price_mean:,.2f}")
        price_median = np.median(price)
        print(f"Median Price: {price_median:,.2f}")
        std_price = np.std(price)
        print(f"Standard Deviation Price: {std_price:,.2f}")
        minimum_price = np.min(price)
        print(f"Minimum Price: {minimum_price:,.2f}")
        maximum_price = np.max(price)
        print(f"Maximum Price: {maximum_price:,.2f}")
        price_range = maximum_price - minimum_price
        print(f"Range Price: {price_range:,.2f}")
        print("Pandas Describe")
        print(self.df.describe())


    def run(self):
        self.load_data()
        self.dataset_info()
        self.clean_data()
        self.basic_statistics()

if __name__ == "__main__":
    analysis = HousePriceAnalysis("datasets/house_prices.csv")
    print("======This is House Price Analysis Project======")
    analysis.run()