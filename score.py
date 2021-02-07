import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model
from sklearn.externals import joblib
import azureml.train.automl

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best-automl.pkl')
    
    model = joblib.load(model_path)

def run(data):
    try:
		
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)

        return result.tolist()
		
    except Exception as e:
	
        error = str(e)
        return error
