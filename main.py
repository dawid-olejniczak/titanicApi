from fastapi import FastAPI
import pandas as pd
from training import ModelTrainer, DataCleaner

app = FastAPI()

# Load and preprocess the data
train_df = pd.read_csv("train.csv")
behavioural_df = train_df[['NObeyesdad', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'CALC', 'SCC', 'FAF', 'TUE', 'MTRANS']]
preprocessed_df = DataCleaner.preprocess_data(behavioural_df)
trainer = ModelTrainer(data=preprocessed_df, target_name='NObeyesdad')
trainer.prepare_data()
results = trainer.train_and_evaluate()

@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.get('/age/{age:int}')
async def get_age(age: int):
    return {"You are this old": age}

@app.get('/predict')
async def predict(model: str, favc: int, fcvc: int, ncp: int, caec: int, ch2o: int,
                  calc: int, scc: int, faf: int, tue: int, mtrans: int):
    data_dict = {'FAVC': [favc],
                 'FCVC': [fcvc],
                 'NCP': [ncp],
                 'CAEC': [caec],
                 'CH2O': [ch2o],
                 'CALC': [calc],
                 'SCC': [scc],
                 'FAF': [faf],
                 'TUE': [tue],
                 'MTRANS': [mtrans]}
    selected_model = trainer.get_model(model)
    return {"message": selected_model}
    data_df = pd.DataFrame(data_dict)
    prediction = selected_model.predict(data_df)
    prediction_numeric = prediction.tolist()[0]

    obesity_scale = {0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Overweight_Level_I', 3: 'Overweight_Level_II',
                     4: 'Obesity_Type_I', 5: 'Obesity_Type_II', 6: 'Obesity_Type_III'}

    prediction_result = {'Your prediction': obesity_scale[prediction_numeric]}

    return prediction_result
