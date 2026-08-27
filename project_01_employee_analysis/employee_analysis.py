import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

class EmployeeAnalysisFramework:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # Load Dataset
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return {
            "Message":"Dataset Loaded Successfully",
            "Rows" : self.df.shape[0],
            "Columns": self.df.shape[1]
        }

    # Dataset Information
    def data_information(self):
        return {
            "columns": self.df.columns.tolist(),
            "Shape": {"Rows": self.df.shape[0],
                      "Columns": self.df.shape[1]},
            "First_5_Rows": self.df.head().to_dict(orient="records")
        }

    # Data Cleaning
    def clean_data(self):
        missing_values = (self.df.isnull().sum().to_dict())
        duplicate_rows = int(self.df.duplicated().sum())
        self.df.drop_duplicates(inplace=True)
        return {
            "Missing Values": missing_values,
            "Duplicate Rows": duplicate_rows,
            "Row After Cleaning": self.df.shape[0]
        }

    # NumPy Analysis
    def numpy_analysis(self):
        salary = self.df["Salary"].values
        average_salary = np.mean(salary)
        median_salary = np.median(salary)
        standard_deviation = np.std(salary)
        self.df["Salary_Level"] = np.where(self.df["Salary"] >= 70000, "High Salary", "Low Salary")
        conditions = [
            self.df["Experience"] < 3,
            self.df["Experience"] <= 8,
            self.df["Experience"] > 8
        ]
        choice = ["Junior", "Mid Level", "Senior"]
        self.df["Experience_Level"] = np.select(conditions, choice, default="Unknown")
        return {
            "average_salary": float(average_salary),
            "median_salary": float(median_salary),
            "standard_deviation": float(standard_deviation)
        }

    # Feature Engineering
    def feature_engineering(self):
        self.df["Salary_Per_Year"] = (self.df["Salary"] / (self.df["Experience"] + 1))
        self.df["Age_Group"] = pd.cut(self.df["Age"], bins=[0, 25, 35, 100], labels=["Young", "Adult", "Senior"])
        return{
            "new_features":["Salary_Per_Year", "Age_Group"]
        }

    # Pandas Analysis
    def pandas_analysis(self):
        department_count = self.df["Department"].value_counts().to_dict()
        department_salary = self.df.groupby("Department")["Salary"].mean().sort_values(ascending=False).to_dict()
        top5_salary = self.df.sort_values(by="Salary",ascending=False).head(5).to_dict(orient="records")
        it_employees = self.df[self.df["Department"] == "IT"].to_dict(orient="records")
        return {
            "Department_count": department_count,
            "Department_salary": department_salary,
            "Top5_Salary": top5_salary,
            "It_Employees": it_employees
        }

    # Visualization
    def visualization(self):
        output = "outputs"
        os.makedirs(output, exist_ok=True)
        sns.set_theme()
        # Bar Chart
        plt.figure(figsize=(8,5))
        (self.df.groupby("Department")["Salary"].mean().plot(kind="bar"))
        plt.title("Average Salary By Department")
        plt.ylabel("Salary")
        plt.tight_layout()
        plt.savefig(f"{output}/department_salary.png")
        plt.close()
        # Pie Chart
        plt.figure(figsize=(7,7))
        self.df["Department"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title("Department Distribution")
        plt.ylabel("")
        plt.savefig(f"{output}/department_pie.png")
        plt.close()
        # Histogram
        plt.figure(figsize=(8,5))
        sns.histplot(self.df["Salary"], bins=8, kde=True)
        plt.title("Salary Distribution")
        plt.savefig(f"{output}/salary_distribution.png")
        plt.close()
        # Box Plot
        plt.figure(figsize=(8,5))
        sns.boxplot(y=self.df["Salary"])
        plt.title("Salary Outlier Detection")
        plt.savefig(f"{output}/salary_boxplot.png")
        plt.close()
        # Scatter Plot
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=self.df, x="Experience", y="Salary", hue="Department")
        plt.title("Experience vs Salary")
        plt.savefig(f"{output}/experience_salary.png")
        plt.close()
        return{
            "Message":"All Charts Saved Successfully",
            "output_directory":f"{output}"
        }


    # Complete Pipeline
    def run(self):
        self.load_data()
        self.data_information()
        self.clean_data()
        self.numpy_analysis()
        self.feature_engineering()
        self.pandas_analysis()
        self.visualization()
        print("\nEmployee Analysis Completed!")

def main():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.run()

if __name__ == "__main__":
    main()