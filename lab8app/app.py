from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import numpy as np
import os

MLFLOW_MODEL_URI = "runs:/582504fe9a7047a29bd8931daa94a4a7/rf_model_max_depth-20_min_samples_leaf-3_min_samples_split-5_n_estimators-100"

model = None

model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)

app = FastAPI()
class PredictionRequest(BaseModel):
    HighBP: float
    HighChol: float
    CholCheck: float
    BMI: float
    Smoker: float
    Stroke: float
    HeartDiseaseorAttack: float
    PhysActivity: float
    Fruits: float
    Veggies: float
    HvyAlcoholConsump: float
    AnyHealthcare: float
    NoDocbcCost: float
    MentHlth: float
    PhysHlth: float
    DiffWalk: float
    Sex: float
    GenHlth_1_0: float
    GenHlth_2_0: float
    GenHlth_3_0: float
    GenHlth_4_0: float
    GenHlth_5_0: float 
    Age_1_0: float     
    Age_2_0: float     
    Age_3_0: float     
    Age_4_0: float    
    Age_5_0: float    
    Age_6_0: float   
    Age_7_0: float     
    Age_8_0: float     
    Age_9_0: float     
    Age_10_0: float 
    Age_11_0: float  
    Age_12_0: float   
    Age_13_0: float    
    Education_1_0: float 
    Education_2_0: float 
    Education_3_0: float 
    Education_4_0: float 
    Education_5_0: float 
    Education_6_0: float 
    Income_1_0: float   
    Income_2_0: float    
    Income_3_0: float  
    Income_4_0: float    
    Income_5_0: float 
    Income_6_0: float   
    Income_7_0: float   
    Income_8_0: float   


@app.get("/")
def read_root():
    return {"message": "Model scoring API is running!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Cannot make predictions.")
    
    try:
        request_dict = request.model_dump()

        original_column_names = {
            'GenHlth_1_0': 'GenHlth_1.0', 'GenHlth_2_0': 'GenHlth_2.0', 'GenHlth_3_0': 'GenHlth_3.0',
            'GenHlth_4_0': 'GenHlth_4.0', 'GenHlth_5_0': 'GenHlth_5.0',
            'Age_1_0': 'Age_1.0', 'Age_2_0': 'Age_2.0', 'Age_3_0': 'Age_3.0', 'Age_4_0': 'Age_4.0',
            'Age_5_0': 'Age_5.0', 'Age_6_0': 'Age_6.0', 'Age_7_0': 'Age_7.0', 'Age_8_0': 'Age_8.0',
            'Age_9_0': 'Age_9.0', 'Age_10_0': 'Age_10.0', 'Age_11_0': 'Age_11.0', 'Age_12_0': 'Age_12.0',
            'Age_13_0': 'Age_13.0',
            'Education_1_0': 'Education_1.0', 'Education_2_0': 'Education_2.0', 'Education_3_0': 'Education_3.0',
            'Education_4_0': 'Education_4.0', 'Education_5_0': 'Education_5.0', 'Education_6_0': 'Education_6.0',
            'Income_1_0': 'Income_1.0', 'Income_2_0': 'Income_2.0', 'Income_3_0': 'Income_3.0', 'Income_4_0': 'Income_4.0',
            'Income_5_0': 'Income_5.0', 'Income_6_0': 'Income_6.0', 'Income_7_0': 'Income_7.0', 'Income_8_0': 'Income_8.0'
        }

        input_data_dict = {original_column_names.get(k, k): v for k, v in request_dict.items()}

        input_df = pd.DataFrame([input_data_dict])

        prediction = model.predict(input_df)

        prediction_output = prediction[0] if isinstance(prediction, (list, np.ndarray, pd.Series, pd.DataFrame)) else prediction

        if isinstance(prediction_output, (float, int)):
             prediction_output = float(prediction_output) if isinstance(prediction_output, float) else int(prediction_output)

        return {"prediction": prediction_output}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
