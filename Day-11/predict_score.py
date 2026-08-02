import joblib
import pandas as pd

# Load the trained model
model = joblib.load("student_score_model.pkl")

print("=" * 50)
print("      Student Score Prediction System")
print("=" * 50)

hours = float(input("Enter Hours Studied: "))
attendance = float(input("Enter Attendance (%): "))
previous_score = float(input("Enter Previous Score: "))

student = pd.DataFrame({
    "Hours_Studied": [hours],
    "Attendance": [attendance],
    "Previous_Score": [previous_score]
})

prediction = model.predict(student)

print("\nPredicted Final Score:", round(prediction[0], 2))