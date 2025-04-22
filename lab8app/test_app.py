import requests
import json

API_URL = "http://127.0.0.1:8000/predict/"

test_data = {
    'HighBP': 1.0,
    'HighChol': 1.0,
    'CholCheck': 1.0,
    'BMI': 0.09302325581395349,
    'Smoker': 1.0,
    'Stroke': 0.0,
    'HeartDiseaseorAttack': 0.0,
    'PhysActivity': 0.0,
    'Fruits': 0.0,
    'Veggies': 0.0,
    'HvyAlcoholConsump': 0.0,
    'AnyHealthcare': 1.0,
    'NoDocbcCost': 0.0,
    'MentHlth': 0.13333333333333333,
    'PhysHlth': 1.0,
    'DiffWalk': 1.0,
    'Sex': 0.0,
    'GenHlth_1_0': 0.0,
    'GenHlth_2_0': 0.0,
    'GenHlth_3_0': 0.0, 
    'GenHlth_4_0': 0.0,
    'GenHlth_5_0': 1.0, 
    'Age_1_0': 0.0,    
    'Age_2_0': 0.0,   
    'Age_3_0': 0.0,     
    'Age_4_0': 0.0,  
    'Age_5_0': 0.0,   
    'Age_6_0': 0.0,   
    'Age_7_0': 0.0,    
    'Age_8_0': 0.0,   
    'Age_9_0': 0.0, 
    'Age_10_0': 0.0,   
    'Age_11_0': 0.0,   
    'Age_12_0': 1.0,    
    'Age_13_0': 0.0,    
    'Education_1_0': 0.0, 
    'Education_2_0': 0.0,
    'Education_3_0': 0.0, 
    'Education_4_0': 1.0, 
    'Education_5_0': 0.0, 
    'Education_6_0': 0.0, 
    'Income_1_0': 0.0,  
    'Income_2_0': 0.0,    
    'Income_3_0': 0.0,   
    'Income_4_0': 0.0,    
    'Income_5_0': 1.0,   
    'Income_6_0': 0.0,   
    'Income_7_0': 0.0,   
    'Income_8_0': 0.0    
}


try:
    response = requests.post(API_URL, json=test_data)

    print(f"Status Code: {response.status_code}")

    try:
        print("Response Body:")
        print(json.dumps(response.json(), indent=4))
    except json.JSONDecodeError:
        print("Could not decode JSON response body.")
        print("Response Text:")
        print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"Error: Could not connect to the API at {API_URL}")
    print("Please ensure your FastAPI application is running.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")