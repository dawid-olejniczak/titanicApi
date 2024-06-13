import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

class DataCleaner:
    @staticmethod
    def preprocess_data(df):
        # Define the mappings for replacement
        replace_mappings = {
            'family_history_with_overweight': {'no': 0, 'yes': 1},
            'FAVC': {'no': 0, 'yes': 1},
            'SMOKE': {'no': 0, 'yes': 1},
            'SCC': {'no': 0, 'yes': 1},
            'CAEC': {'no': 3, 'Sometimes': 2, 'Frequently': 1, 'Always': 0},
            'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'MTRANS': {'Walking': 0, 'Bike': 1, 'Motorbike': 2, 'Automobile': 3, 'Public_Transportation': 4},
            'Gender': {'Male': 0, 'Female': 1},
            'NObeyesdad': {'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 'Overweight_Level_II': 3,
         'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6}
        }

        # Replace values in the DataFrame based on the mappings, if the column is present
        for column, mapping in replace_mappings.items():
            if column in df.columns:
                df[column] = df[column].replace(mapping)

        # Calculate BMI if 'Weight' and 'Height' columns are present
        if 'Weight' in df.columns and 'Height' in df.columns:
            df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

        # Select only numeric columns
        res_df = df.select_dtypes(include=[np.number])

        return res_df

class ModelTrainer:
    def __init__(self, data, target_name):
        self.data = data
        self.target_name = target_name
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "GaussianNaiveBayes": GaussianNB(),
            "StochasticGradientDescent": SGDClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        }
        self.results = {}

    def get_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(self.models.keys())}")

    def prepare_data(self):
        X = self.data.drop([self.target_name], axis=1)
        y = self.data[self.target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def train_and_evaluate(self):
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[name] = accuracy
        return self.results

