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