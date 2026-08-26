from fastapi import FastAPI
from employee_analysis import EmployeeAnalysisFramework

app = FastAPI(
    title="Employee Analysis API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Employee Analysis API is Running..."}

@app.get("/employees")
def get_employees():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    information = framework.data_information()
    return information

@app.get("/employees/clean")
def clean_employees():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    cleaning_result = framework.clean_data()
    return cleaning_result

@app.get("/employees/numpy")
def numpy_analysis():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    numpy_result = framework.numpy_analysis()
    return numpy_result

@app.get("/employees/features")
def features_engineering():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    features_result = framework.feature_engineering()
    return features_result