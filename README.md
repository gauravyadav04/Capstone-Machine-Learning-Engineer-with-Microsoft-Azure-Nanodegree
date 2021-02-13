*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Predict mortality caused by heart failure

This project is part of the Udacity Azure ML Nanodegree. In this project, I have used [heart failure clinical records dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) to train a machine learning model using Hyperdrive and AutoML API from AzureML and deploy the best model.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset - Heart failure clinical dataset

### Overview
The dataset contains 13 features, which report clinical, body, and lifestyle information

| Feature | Description | Measurement |
| ------ | ------ | ------ |
| age | Age of the patient| Years |
| anaemia | Decrease of red blood cells or hemoglobin | Boolean |
| creatinine phosphokinase | Level of the CPK enzyme in the blood | mcg/L |
| diabetes | If the patient has diabetes | Boolean |
| ejection fraction | Percentage of blood the left ventricle pumps out with each contraction | Percentage |
| high blood pressure | If a patient has hypertension | Boolean |
| platelets | Platelets in the blood | kiloplatelets/mL |
| serum creatinine | Level of creatinine in the blood | mcg/dL |
| serum sodium | Level of sodium in the blood | mEq/L |
| sex | Gender of patient | Binary |
| smoking | If the patient smokes | Boolean |
| time | Follow-up period | Days |
| (target) death event | If the patient died during the follow-up period | Boollean |

### Task
The objective is to train a mchine learning model to predict the target variable - death event - (0 - patient survived and 1 - patient deceased)

### Access
I have downloaded the data from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) 
* uploaded and registered the dataset in Azure ML Studio to access in workspace
* uploaded to my [github](https://raw.githubusercontent.com/gauravyadav04/Capstone-Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree/main/data/heart_failure_clinical_records_dataset.csv) to access it in train.py 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
I have used logistic regression to predict the target variable, logistic regression is easier to implement, interpret, and very efficient to train. I have used following parameters -
* Regularization parameter C, range used [0.01,0.1,1]
* Maximum number of iterations max_iter, range used [50, 100, 150, 200]


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

![2](https://user-images.githubusercontent.com/6285945/107850301-6a3b8e80-6e27-11eb-9089-2921431ec4cb.png)

![4](https://user-images.githubusercontent.com/6285945/107850303-6b6cbb80-6e27-11eb-9e25-a4f6f5b62ec5.png)

![5](https://user-images.githubusercontent.com/6285945/107850305-6dcf1580-6e27-11eb-9ef8-af689b11be49.png)

![6](https://user-images.githubusercontent.com/6285945/107850307-6e67ac00-6e27-11eb-8e50-c27fd65f0cd1.png)

![3](https://user-images.githubusercontent.com/6285945/107850302-6ad42500-6e27-11eb-8e56-b77837e1a6cd.png)

![7](https://user-images.githubusercontent.com/6285945/107850309-6f98d900-6e27-11eb-962e-a537f3cb05c2.png)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

