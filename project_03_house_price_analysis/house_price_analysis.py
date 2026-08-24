from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class HousePriceAnalysis:
    def __init__(self,file_path):
        self.file_path = Path(file_path)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok = True)

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
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
                median_value = self.df[column].median()
                self.df[column] = (self.df[column].fillna(median_value))
                print(f"Filled Missing Values:{column}")
        #Handle Missing Categorical values
        categorical_columns = ["Location","Property_Type"]
        for column in categorical_columns:
            if self.df[column].isnull().any():
                mode_value = (self.df[column].mode()[0])
                self.df[column] = self.df[column].fillna(mode_value)
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
        print("Data Cleaning Completed")

    #Basic Statistic
    def basic_statistics(self) -> None:
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        print(f"\n===PRICE STATISTICS===")
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

    def price_distribution(self) -> None:
        print(f"\n===PRICE DISTRIBUTION===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        price = self.df["Price"].to_numpy()
        print(f"Price Statistics...")
        print(f"Mean: {np.mean(price):.2f}")
        print(f"Median: {np.median(price):.2f}")
        print(f"Standard Deviation: {np.std(price):.2f}")
        print(f"Minimum: {np.min(price):.2f}")
        print(f"Maximum: {np.max(price):.2f}")
        print(f"Std Deviation: {np.std(price):.2f}")
        plt.figure(figsize=(10,6))
        sns.histplot(data=self.df,x="Price",bins=15,kde=True)
        plt.title("House Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Number of Houses")
        plt.tight_layout()
        plt.savefig(self.output_dir/"price_distribution.png",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close()
        print("***Price Distribution Chart Saved***")

    def detect_outliers(self) -> None:
        print(f"\n===OUTLIERS DETECTION===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        price = self.df["Price"]
        q1 = price.quantile(0.25)
        q3 = price.quantile(0.75)
        iqr = q3 - q1
        lower_bound = (q1 - 1.5 * iqr)
        upper_bound = (q3 + 1.5 * iqr)
        print(f"Q1:{q1:,.2f}")
        print(f"Q3:{q3:,.2f}")
        print(f"IQR:{iqr:,.2f}")
        print(f"Lower Bound:{lower_bound:,.2f}")
        print(f"Upper Bound:{upper_bound:,.2f}")
        outlier_condition = ((price < lower_bound) | (price > upper_bound))
        self.outliers = self.df[outlier_condition].copy()
        print(f"Number of Outliers:{len(self.outliers)}")
        if len(self.outliers) > 0:
            print(f"Outlier Houses")
            print(self.outliers[["House_ID","Area","Bedrooms","Location","Property_Type","Price"]].sort_values(by="Price",ascending=False))
        else:
            print(f"\n No Price Outlier Detected")
        plt.figure(figsize=(10,5))
        sns.boxplot(data=self.df,x="Price")
        plt.title("House Price Outlier Detection")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.savefig(self.output_dir/"price_boxplot.png",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close()
        print("Outlier box plot saved")

    def outlier_summary(self) -> None:
        print(f"\n===OUTLIER SUMMARY===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        if not hasattr(self,"outliers"):
            self.detect_outliers()
        if self.outliers.empty:
            print(f"No outliers available")
            return
        print(f"\nOutlier by Location")
        print(self.outliers["Location"].value_counts())
        print(f"\nOutliers by Property Type")
        print(self.outliers["Property_Type"].value_counts())
        average_outlier_price = (self.outliers["Price"].mean())
        print(f"\nAverage Outlier Price:{average_outlier_price:,.2f}")
        highest_price_house = (self.outliers.sort_values(by="Price",ascending=False).iloc[0])
        print("\nHighest Price Outlier")
        print(f"House ID:{highest_price_house['House_ID']}")
        print(f"Area:{highest_price_house['Area']}")
        print(f"Location:{highest_price_house['Location']}")
        print(f"Price:{highest_price_house['Price']:,.2f}")

    def area_price_analysis(self) -> None:
        print(f"\n===Area Price Analysis===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        area_price_corr = (self.df[["Area","Price"]].corr().loc["Area","Price"])
        print("Area Price Analysis:{area_price_corr:,.4f}")
        print("\nAverage Price by Property Type")
        property_price = (self.df.groupby("Property_Type")["Price"].mean().sort_values(ascending=False))
        print(property_price)
        print("\nAverage Price by Location")
        location_price = (self.df.groupby("Location")["Price"].mean().sort_values(ascending=False))
        print(location_price)
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=self.df,x="Area",y="Price",hue="Property_Type",style="Location",s=100)
        plt.title("Area VS House Price Analysis")
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()
        plt.savefig(self.output_dir/"area_price_analysis.png",dpi=300,bbox_inches="tight")
        plt.close()
        print("\nArea VS Price Chart Saved")

    def correlation_analysis(self) -> None:
        print(f"\n===CORRELATION ANALYSIS===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        numerical_df = self.df.select_dtypes(include=np.number)
        correlation_matrix = numerical_df.corr()
        print("\nCorrelation Matrix")
        print(correlation_matrix.round(3))
        price_correlation = (correlation_matrix["Price"].sort_values(ascending=False))
        print(f"\nCorrelation with Price")
        print(price_correlation.round(3))
        positive_features = (price_correlation[price_correlation > 0.5])
        print(f"\nStrong Positive Correlation")
        print(positive_features.round(3))
        negative_features = (price_correlation[price_correlation < -0.5])
        print(f"Strong Negative Correlation")
        print(negative_features.round(3))

    def correlation_heatmap(self) -> None:
        print(f"\n===CORRELATION ANALYSIS===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        numerical_df = self.df.select_dtypes(include=np.number)
        correlation_matrix =(numerical_df.corr())
        plt.figure(figsize=(12,8))
        sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm",linewidths=0.5)
        plt.title("House Price Correlation Matrix")
        plt.tight_layout()
        plt.show()
        plt.savefig(self.output_dir/"correlation_heatmap.png",dpi=300, bbox_inches="tight")
        plt.close()
        print("Correlation Heatmap Saved")

    def feature_engineering(self) -> None:
        print(f"\n===Feature Engineering ANALYSIS===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        self.df["Price_Per_Area"] = (self.df["Price"]/self.df["Area"])
        self.df["Total_Rooms"] = self.df["Bedrooms"] + self.df["Bathrooms"]
        self.df["Area_per_Bedrooms"] = self.df["Area"] / self.df["Bedrooms"]
        age_conditions = [self.df["Age"] <= 5,
                          self.df["Age"] <= 10,
                          self.df["Age"] <= 20,
                          self.df["Age"] > 20]
        age_choices = ["New","Modern","Old","Very Old"]
        self.df["Age_Group"] = np.select(age_conditions, age_choices, default="Unknown")
        area_conditions = [self.df["Area"] < 1000,
                           self.df["Area"] < 1500,
                           self.df["Area"] < 2000,
                           self.df["Area"] >= 2000]
        area_choices = ["Small","Median","Large","Very Large"]
        self.df["Area_Category"] = np.select(area_conditions, area_choices, default="Unknown")
        self.df["Has_Parking"] = np.where(self.df["Parking"] > 0,"Yes","No")
        new_features=["Price_Per_Area",
                      "Total_Rooms",
                      "Area_per_Bedrooms",
                      "Age_Group",
                      "Area_Category",
                      "Has_Parking"]
        print("\nNew Features")
        print(self.df[new_features].head())
        print(f"Feature Engineering Completed.")


    def feature_analysis(self) -> None:
        print(f"\n===Feature Engineering ANALYSIS===")
        if self.df is None:
            raise ValueError("Dataset has not been loaded")
        print(f"\nAverage Price By Location")
        location_analysis = (self.df.groupby("Location")["Price"].agg(["mean","median","count"]).sort_values(by="mean",ascending=False))
        print(location_analysis.round(2))
        print("\nAverage Price By Property Type")
        property_analysis = (self.df.groupby("Property_Type")["Price"].agg(["mean","median","count"]).sort_values(by="mean",ascending=False))
        print(property_analysis.round(2))
        print("\nAverage Price By Age Group")
        age_analysis = (self.df.groupby("Age_Group",observed=True)["Price"].mean().sort_values(ascending=False))
        print(f"\nAverage Price By Area Category")
        area_analysis = (self.df.groupby("Area_Category")["Price"].mean().sort_values(ascending=False))
        print(area_analysis.round(2))
        print(f"\nAverage Price By Parking Availability")
        parking_analysis = (self.df.groupby("Has_Parking")["Price"].mean().sort_values(ascending=False))
        print(parking_analysis.round(2))

    def run(self):
        self.load_data()
        self.dataset_info()
        self.clean_data()
        self.basic_statistics()
        self.price_distribution()
        self.detect_outliers()
        self.outlier_summary()
        self.area_price_analysis()
        self.correlation_analysis()
        self.correlation_heatmap()
        self.feature_engineering()
        self.feature_analysis()

if __name__ == "__main__":
    analysis = HousePriceAnalysis("datasets/house_prices.csv")
    print("======This is House Price Analysis Project======")
    analysis.run()