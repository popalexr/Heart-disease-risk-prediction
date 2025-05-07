import joblib
import pandas as pd
from tkinter import *
from tkinter import messagebox

class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction")
        self.root.geometry("400x600")

        # Load pretrained logistic regression model
        self.model = joblib.load('logistic_regression_model.pkl')

        self.create_widgets()

    def validate_age(self, P):
        # allow only numeric input
        if P.isdigit():
            return True
        return False

    def create_widgets(self):
        Label(self.root, text="Enter Patient Data", font=("Arial", 14)).pack(pady=10)

        # Age entry (numeric only)
        Label(self.root, text="Age:").pack(anchor='w', padx=20)
        vcmd = (self.root.register(self.validate_age), '%P')
        self.age_entry = Entry(self.root, validate='key', validatecommand=vcmd)
        self.age_entry.pack(padx=20, pady=5)

        # Binary features as checkboxes
        self.chest_pain_var = IntVar()
        Checkbutton(self.root, text="Chest Pain", variable=self.chest_pain_var).pack(anchor='w', padx=20)

        self.shortness_var = IntVar()
        Checkbutton(self.root, text="Shortness of Breath", variable=self.shortness_var).pack(anchor='w', padx=20)

        self.fatigue_var = IntVar()
        Checkbutton(self.root, text="Fatigue", variable=self.fatigue_var).pack(anchor='w', padx=20)

        self.palpitations_var = IntVar()
        Checkbutton(self.root, text="Palpitations", variable=self.palpitations_var).pack(anchor='w', padx=20)

        self.dizziness_var = IntVar()
        Checkbutton(self.root, text="Dizziness", variable=self.dizziness_var).pack(anchor='w', padx=20)

        self.swelling_var = IntVar()
        Checkbutton(self.root, text="Swelling", variable=self.swelling_var).pack(anchor='w', padx=20)

        self.pain_arms_jaw_back_var = IntVar()
        Checkbutton(self.root, text="Pain in Arms, Jaw, or Back", variable=self.pain_arms_jaw_back_var).pack(anchor='w', padx=20)

        self.cold_sweats_nausea_var = IntVar()
        Checkbutton(self.root, text="Cold Sweats / Nausea", variable=self.cold_sweats_nausea_var).pack(anchor='w', padx=20)

        self.high_bp_var = IntVar()
        Checkbutton(self.root, text="High Blood Pressure", variable=self.high_bp_var).pack(anchor='w', padx=20)

        self.high_cholesterol_var = IntVar()
        Checkbutton(self.root, text="High Cholesterol", variable=self.high_cholesterol_var).pack(anchor='w', padx=20)

        self.diabetes_var = IntVar()
        Checkbutton(self.root, text="Diabetes", variable=self.diabetes_var).pack(anchor='w', padx=20)

        self.smoking_var = IntVar()
        Checkbutton(self.root, text="Smoking", variable=self.smoking_var).pack(anchor='w', padx=20)

        self.obesity_var = IntVar()
        Checkbutton(self.root, text="Obesity", variable=self.obesity_var).pack(anchor='w', padx=20)

        self.sedentary_var = IntVar()
        Checkbutton(self.root, text="Sedentary Lifestyle", variable=self.sedentary_var).pack(anchor='w', padx=20)

        self.chronic_stress_var = IntVar()
        Checkbutton(self.root, text="Chronic Stress", variable=self.chronic_stress_var).pack(anchor='w', padx=20)

        self.gender_var = IntVar()
        Checkbutton(self.root, text="Is female", variable=self.gender_var).pack(anchor='w', padx=20)

        self.family_history_var = IntVar()
        Checkbutton(self.root, text="Family History of Heart Disease", variable=self.family_history_var).pack(anchor='w', padx=20)

        self.predict_button = Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=20)

    def predict(self):
        try:
            age = float(self.age_entry.get())
            chest_pain = self.chest_pain_var.get()
            shortness = self.shortness_var.get()
            fatigue = self.fatigue_var.get()
            palpitations = self.palpitations_var.get()
            dizziness = self.dizziness_var.get()
            swelling = self.swelling_var.get()
            pain_arms_jaw_back = self.pain_arms_jaw_back_var.get()
            cold_sweats_nausea = self.cold_sweats_nausea_var.get()
            high_bp = self.high_bp_var.get()
            high_cholesterol = self.high_cholesterol_var.get()
            diabetes = self.diabetes_var.get()
            smoking = self.smoking_var.get()
            obesity = self.obesity_var.get()
            sedentary = self.sedentary_var.get()
            chronic_stress = self.chronic_stress_var.get()
            gender = self.gender_var.get()
            family_history = self.family_history_var.get()
            feature_names = [
                'Chest_Pain',
                'Shortness_of_Breath',
                'Fatigue',
                'Palpitations',
                'Dizziness',
                'Swelling',
                'Pain_Arms_Jaw_Back',
                'Cold_Sweats_Nausea',
                'High_BP',
                'High_Cholesterol',
                'Diabetes',
                'Smoking',
                'Obesity',
                'Sedentary_Lifestyle',
                'Family_History',
                'Chronic_Stress',
                'Gender',
                'Age'
            ]
            X = pd.DataFrame([[
                chest_pain,
                shortness,
                fatigue,
                palpitations,
                dizziness,
                swelling,
                pain_arms_jaw_back,
                cold_sweats_nausea,
                high_bp,
                high_cholesterol,
                diabetes,
                smoking,
                obesity,
                sedentary,
                family_history,
                chronic_stress,
                gender,
                age
            ]], columns=feature_names)

            proba = self.model.predict_proba(X)[0, 1]
            label = self.model.predict(X)[0]

            messagebox.showinfo(
                "Result",
                f"Prediction: {'Risk' if label == 1 else 'No Risk'}\nProbability: {proba:.2f}"
            )
        except Exception as ex:
            messagebox.showerror("Input Error", "Please enter valid values.")
            print(ex)

if __name__ == "__main__":
    root = Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()