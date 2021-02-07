from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Workspace

# subscription_id = 'f9d5a085-54dc-4215-9ba6-dad5d86e60a0'
# resource_group = 'aml-quickstarts-137804'
# workspace_name = 'quick-starts-ws-137804'

# workspace = Workspace(subscription_id, resource_group, workspace_name)

# dataset = Dataset.get_by_name(workspace, name='Heart_Failure_Clinical_Records_Dataset')

ws = Workspace.from_config()

dataset = Dataset.get_by_name(ws, name='Heart_Failure_Clinical_Records_Dataset')

df = dataset.to_pandas_dataframe()

df = dataset.to_pandas_dataframe()

x = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    
    #os.makedirs('outputs', exist_ok=True)  
    #joblib.dump(model, 'outputs/model.joblib')
    
    run.log("Accuracy", np.float(accuracy))
if __name__ == '__main__':
    main()
