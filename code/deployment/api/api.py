from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load transformations
with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Define the FastAPI api
api = FastAPI()


# Define the input data schema
class CreditScoreInput(BaseModel):
    revolving_utilization_of_unsecured_lines: float
    age: int
    number_of_time_30_to_59_days_past_due_not_worse: int
    debt_ratio: float
    monthly_income: int
    number_of_open_credit_lines_and_loans: int
    number_of_90_days_late: int
    number_real_estate_loans_or_lines: int
    number_of_time_60_to_89_days_past_due_not_worse: int
    number_of_dependents: int


# Define the prediction endpoint

@api.post("/predict")
def predict(input_data: CreditScoreInput):

    # Preprocess
    total_late_days = input_data.number_of_time_30_to_59_days_past_due_not_worse + \
        input_data.number_of_time_60_to_89_days_past_due_not_worse + \
        input_data.number_of_90_days_late

    income_expense_difference = input_data.monthly_income - input_data.monthly_income * input_data.debt_ratio

    late_90_days_likelihood = (0.1 * input_data.number_of_time_30_to_59_days_past_due_not_worse + 
                               0.2 * input_data.number_of_time_60_to_89_days_past_due_not_worse +
                               0.7 * input_data.number_of_90_days_late)

    data = [[
        input_data.revolving_utilization_of_unsecured_lines,
        input_data.age,
        input_data.number_of_open_credit_lines_and_loans,
        input_data.number_real_estate_loans_or_lines,
        input_data.number_of_dependents,
        total_late_days,
        income_expense_difference,
        late_90_days_likelihood
    ]]

    data = scaler.transform(imputer.transform(data))

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}
