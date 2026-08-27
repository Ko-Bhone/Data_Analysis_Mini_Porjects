from fastapi import FastAPI
from employee_analysis import EmployeeAnalysisFramework
from fastapi.responses import FileResponse


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

@app.get("/employees/pandas")
def pandas_analysis():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    pandas_result = framework.pandas_analysis()
    return pandas_result

@app.get("/employees/charts")
def generate_charts():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    chart_result = framework.visualization()
    return chart_result

@app.get("/employees/charts/department-pie")
def department_pie_chart():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    framework.visualization()
    return FileResponse(
        "outputs/department_pie.png",
        media_type="image/png",
        filename="department_pie.png"
    )

@app.get("/employees/charts/salary-distribution")
def salary_distribution_chart():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    framework.visualization()
    return FileResponse(
        "outputs/salary_distribution.png",
        media_type="image/png",
        filename="salary_distribution.png"
    )

@app.get("/employees/charts/salary-boxplot")
def salary_boxplot_chart():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    framework.visualization()
    return FileResponse(
        "outputs/salary_boxplot.png",
        media_type="image/png",
        filename="salary_boxplot.png"
    )

@app.get("/employees/charts/experience-salary")
def experience_salary_chart():
    framework = EmployeeAnalysisFramework("datasets/employee_data.csv")
    framework.load_data()
    framework.visualization()
    return FileResponse(
        "outputs/experience_salary.png",
        media_type="image/png",
        filename="experience_salary.png"
    )

