import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TitanicAnalysis:
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    # Load Dataset
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print("=" * 25)
            print("Dataset Loaded Successfully")
            print("=" * 25)
        except FileNotFoundError:
            print("File Not Found:{self.file_path}")
            raise

    # Dataset Information
    def data_information(self):
        print("***Dataset Information***")
        print("\nFirst Five Rows")
        print(self.df.head(5))
        print("\nShape")
        print(self.df.shape)
        print("\nColumns")
        print(self.df.columns.tolist())
        print("\nDataset Information")
        print(self.df.info())
        print("\nStatistical Summary Information")
        print(self.df.describe(include='all'))
        print("\nMissing Values")
        print(self.df.isnull().sum())
        print("\nDuplicate Rows")
        print(self.df.duplicated().sum())

    # Data Cleaning
    def clean_data(self):
        print("Cleaning Dataset...")
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        print(f"Removed Duplicate Rows: {before - after}")
        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Embarked"] = self.df["Embarked"].fillna(self.df["Embarked"].mode()[0])
        self.df["Cabin"] = self.df["Cabin"].fillna("Unknown")
        self.df = self.df[self.df["Fare"] >= 0]
        self.df["Survived"] = self.df["Survived"].astype(int)
        self.df["Pclass"] = self.df["Pclass"].astype(int)
        self.df["Age"] = self.df["Age"].astype(float)
        self.df["Fare"] = self.df["Fare"].astype(float)
        print("\nCleaning Completed Dataset...")
        print("\nRemaining Missing Values")
        print("-"*25)
        print(self.df.isnull().sum())

    # Display Basic Statistics
    def basic_statistics(self) -> None:
        print("\nBasic Statistics")
        age = self.df["Age"].to_numpy()
        fare = self.df["Fare"].to_numpy()
        print(f"Average Age:{np.mean(age):.2f}")
        print(f"Median Age:{np.median(age):.2f}")
        print(f"Age std:{np.std(age):.2f}")
        print("*" * 15)
        print(f"Average Fare:{np.mean(fare):.2f}")
        print(f"Median Fare:{np.median(fare):.2f}")
        print(f"Fare std:{np.std(fare):.2f}")

    # Feature Engineering
    def feature_engineering(self):
        print("***Create New Features...***")
        self.df["Family_Size"] = (self.df["SibSp"] + self.df["Parch"] + 1)
        self.df["Is_Alone"] = np.where(self.df["Family_Size"] ==1,"Yes","No")
        self.df["Age_Group"] = pd.cut(self.df["Age"],bins=[0,12,18,35,60,100],labels=["Child","Teen","Adult","Middle Age","Senior"])
        fare_conditions = [self.df["Fare"] < 10,
                           self.df["Fare"] < 30,
                           self.df["Fare"] < 70,
                           self.df["Fare"] >= 70]
        fare_choice = ["Low","Medium","High","Luxury"]
        self.df["Fare_Category"] = np.select(fare_conditions,fare_choice,default="Unknown")
        self.df["Survival_Label"] = np.where(self.df["Survived"] == 1,"Survived","Not Survived")
        print("Feature Engineering Completed Dataset...")
        print("\nNew Columns")
        print(self.df[["Family_Size","Is_Alone","Age_Group","Fare_Category","Survival_Label"]].head(5))

    # Survival Analysis
    def survival_analysis(self):
        print("***Survival Analysis...***")
        survival_rate = (self.df["Survived"].mean() * 100)
        print(f"Overall Survival Rate: {survival_rate:.2f}%")
        print("\nSurvival by Gender")
        gender = (self.df.groupby("Sex")["Survived"].mean() * 100)
        print(gender.round(2))
        print("\nSurvival by Passenger Class")
        pclass = (self.df.groupby("Pclass")["Survived"].mean() * 100)
        print(pclass.round(2))
        print("\nSurvival by Age Group")
        age_group = (self.df.groupby("Age_Group", observed=True)["Survived"].mean() * 100)
        print(age_group.round(2))
        print("\nSurvival by Embarked")
        embarked = (self.df.groupby("Embarked")["Survived"].mean() * 100)
        print(embarked.round(2))
        print("\nAverage Family Size")
        print(self.df["Family_Size"].describe())
        print("\nFare Category Count")
        print(self.df["Fare_Category"].value_counts())
        print("\nTop 10 Highest Fare")
        print(self.df[["Name","Fare","Pclass","Survived"]].sort_values(by="Fare",ascending=False).head(10))

    # Correlation Analysis
    def correlation_analysis(self):
        print("=" * 30)
        print("Correlation Analysis....")
        numeric_df = self.df.select_dtypes(include=np.number)
        correlation = numeric_df.corr(numeric_only = True)
        print("\nCorrelation Matrix")
        print(correlation)
        print("\nCorrelation with Survival")
        print(correlation["Survived"].sort_values(ascending=False))
        print("\nStrongest Positive Correlation")
        print(correlation["Survived"].drop("Survived").idxmax())
        print("\nStrongest Negative Correlation")
        print(correlation["Survived"].drop("Survived").idxmin())

    # Survival Count Plot
    def plot_survival_count(self) -> None:
        print("""Survival Count Plot""")
        plt.figure(figsize=(8,5))
        sns.countplot(data=self.df, x="Survived")
        plt.title("Survival Count")
        plt.xlabel("Survived")
        plt.ylabel("Passenger Count")
        plt.tight_layout()
        plt.show()
        plt.savefig("outputs/survival_count.png",dpi=300)
        plt.close()

    # Survival By Gender
    def plot_survival_by_Gender(self) -> None:
        plt.figure(figsize=(8,5))
        sns.countplot(data=self.df,x="Sex",hue="Survived")
        plt.title("Survival by Gender")
        plt.tight_layout()
        plt.show()
        plt.savefig("outputs/survival_by_gender.png",dpi=300)
        plt.close()

    # Survival By Passenger Class
    def plot_survival_by_class(self) -> None:
        plt.figure(figsize=(8,5))
        sns.countplot(data=self.df,x="Pclass",hue="Survived")
        plt.title("Survival by Passenger Class")
        plt.tight_layout()
        plt.savefig("outputs/survival_by_class.png",dpi=300)
        plt.close()

    # Age Distribution
    def plot_age_Distribution(self) -> None:
        plt.figure(figsize=(8,5))
        sns.histplot(data=self.df["Age"],bins=20,kde=True)
        plt.title("Age Distribution")
        plt.tight_layout()
        plt.savefig("outputs/age_distribution.png",dpi=300)
        plt.close()

    # Fare Box Plot
    def plot_fare_box(self) -> None:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=self.df,y="Fare")
        plt.title("Fare Box Plot")
        plt.tight_layout()
        plt.savefig("outputs/fare_box.png",dpi=300)
        plt.close()

    # Correlation Heatmap
    def plot_heatmap(self) -> None:
        numeric_df = self.df.select_dtypes(include=np.number)
        correlation  = numeric_df.corr(numeric_only = True)
        plt.figure(figsize=(10,8))
        sns.heatmap(correlation, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig("outputs/heatmap.png",dpi=300)
        plt.close()

    # Pair Plot
    def plot_pairplot(self) -> None:
        columns = ["Age","Fare","Pclass","Survived"]
        pair = sns.pairplot(self.df[columns],hue="Survived")
        pair.fig.suptitle("Pair Plot",y=1.02)
        pair.savefig("outputs/pairplot.png",dpi=300)
        plt.close()

    # Survival Rate Bar Plot
    def plot_survival_rate(self) -> None:
        survival = (self.df.groupby("Sex")["Survived"].mean().reset_index())
        plt.figure(figsize=(8,5))
        sns.barplot(data=survival,x="Sex",y="Survived")
        plt.title("Average Survival Rate by Gender")
        plt.ylabel("Survival Rate")
        plt.xlabel("Gender")
        plt.tight_layout()
        plt.savefig("outputs/survival_rate.png",dpi=300)
        plt.close()

    # Age Group Distribution
    def plot_age_group(self) -> None:
        plt.figure(figsize=(8,5))
        sns.countplot(data=self.df,x="Age_Group")
        plt.title("Age Group Distribution")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig("outputs/age_group_distribution.png",dpi=300)
        plt.close()

    # Fare Distribution by Class
    def plot_fare_by_class(self) -> None:
        plt.figure(figsize=(8,5))
        sns.boxplot(data=self.df,x="Pclass",y="Fare")
        plt.title("Fare by Passenger Class")
        plt.tight_layout()
        plt.savefig("outputs/fare_by_class.png",dpi=300)
        plt.close()

    # Run All Visualization
    def visualization(self):
        "Generate All Charts"
        print("Generating All Charts...")
        self.plot_survival_count()
        self.plot_survival_by_Gender()
        self.plot_survival_by_class()
        self.plot_age_Distribution()
        self.plot_fare_box()
        self.plot_heatmap()
        self.plot_pairplot()
        self.plot_survival_rate()
        self.plot_age_group()
        self.plot_fare_by_class()
        print("All Charts Saved successfully")

    # Save Cleaned Dataset
    def save_results(self) -> None:
        output_file = self.output_dir / "cleaned_titanic.csv"
        self.df.to_csv(output_file,index=False)
        print(f"\nCleaned dataset saved to:")
        print(output_file)

    # Final Summary
    def final_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PROJECT SUMMARY")
        print("=" * 60)
        total = len(self.df)
        survived = self.df["Survived"].sum()
        not_survived = total - survived
        survival_rate = (survived / total) * 100
        print(f"Total Passengers      : {total}")
        print(f"Survived              : {survived}")
        print(f"Not Survived          : {not_survived}")
        print(f"Survival Rate         : {survival_rate:.2f}%")
        print("\nAverage Age")
        print(f"{self.df['Age'].mean():.2f}")
        print("\nAverage Fare")
        print(f"{self.df['Fare'].mean():.2f}")
        print("\nPassengers By Class")
        print(self.df["Pclass"].value_counts().sort_index())
        print("\nPassengers By Gender")
        print(self.df["Sex"].value_counts())
        print("\nFeature Columns")
        print(["Family_Size", "Is_Alone", "Age_Group", "Fare_Category", "Survival_Label"])
        print("\nCharts saved in outputs/ folder")
        print("=" * 60)


    # Run Project
    def run(self) -> None:
        self.load_data()
        self.data_information()
        self.clean_data()
        self.basic_statistics()
        self.feature_engineering()
        self.survival_analysis()
        self.correlation_analysis()
        self.visualization()
        self.save_results()
        self.final_summary()

if __name__ == "__main__":
    analysis = TitanicAnalysis("datasets/titanic.csv")
    analysis.run()