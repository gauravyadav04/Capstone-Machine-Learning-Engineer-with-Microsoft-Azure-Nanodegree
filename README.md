# Predict mortality caused by heart failure

This project is part of the Udacity Azure ML Nanodegree. In this project, I have used [heart failure clinical records dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). The objective is to train a machine learning model using Hyperdrive and AutoML and predict mortality caused by heart failure. Models from these tow experiments are compared using Accuracy score, best model is registered and deployed to Azure Container Service as a REST endpoint with key based authentication.


## Dataset - Heart failure clinical dataset

### Overview
The dataset contains detials of 299 patients - 105 women and 194 men - there are 13 features, which report clinical, body, and lifestyle information

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
The objective is to train and deploy a machine learning model which predicts the target variable - death event - (0 - patient survived and 1 - patient deceased)

### Access
I have downloaded the data from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) 
* uploaded and registered the dataset in Azure ML Studio to access in workspace
* uploaded to my [github](https://raw.githubusercontent.com/gauravyadav04/Capstone-Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree/main/data/heart_failure_clinical_records_dataset.csv) to access it in train.py 

## Automated ML
Overview of the `automl` settings and configuration used for this experiment

AutoML Settings

| Name | Description | Value |
| ------ | ------ | ------ |
| experiment_timeout_minutes | Defines as how long experement will run | 30 |
| max_concurrent_iterations | Represents the maximum number of iterations that would be executed in parallel. The default value is 1 | 4 |
| n_cross_validations | Number of cross validations to perform | 5 |
| primary_metric | The metric that Automated Machine Learning will optimize for model selection | "accuracy" |

AutoML Config

| Name | Description | Value |
| ------ | ------ | ------ |
| compute_target | The Azure Machine Learning compute target to run the Automated Machine Learning experiment on | "capstone-cluster" |
| task | Type of task to run | "classification" |
| training_data | dataset to be trained on | dataset |
| label_column_name | Coumn to be predicted | "DEATH_EVENT" |
| path | Folder path | "./automl" |
| enable_early_stopping | Early termination if the score is not improving in the short term | True |
| enable_onnx_compatible_models | Enable or disable enforcing the ONNX-compatible models | True |
| featurization | FeaturizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used | "auto" |
| debug_log | The log file to write debug information to | "automl_errors.log" |

The below screenshot shows the `automl` settings and configuration used for this experiment

![0](https://user-images.githubusercontent.com/6285945/107850354-c1416380-6e27-11eb-9f1c-26a4f210a70a.png)

### Results
The following screenshots show the successfully completed AutoML run. The best model from this experiment was the VotingEnsemble with an `Accuracy` of 86.9%

![1](https://user-images.githubusercontent.com/6285945/107850357-c3a3bd80-6e27-11eb-89a0-8181dc4eb8e6.png)

![2](https://user-images.githubusercontent.com/6285945/107850359-c56d8100-6e27-11eb-9b41-cc669a10c2e9.png)

Below are the screenshots of `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![3](https://user-images.githubusercontent.com/6285945/107850362-c7374480-6e27-11eb-9b0f-915a55dd1f41.png)

![4](https://user-images.githubusercontent.com/6285945/107850365-c9999e80-6e27-11eb-9ca8-72bd253d45e8.png)

![5](https://user-images.githubusercontent.com/6285945/107850367-cc948f00-6e27-11eb-8ad4-8c088ecc09b2.png)

![6](https://user-images.githubusercontent.com/6285945/107850368-ce5e5280-6e27-11eb-846a-06660d6e341b.png)

![7](https://user-images.githubusercontent.com/6285945/107850372-d0c0ac80-6e27-11eb-83cb-0a51d40d6a4f.png)

Saved and registered the best model - 

![9](https://user-images.githubusercontent.com/6285945/107850373-d28a7000-6e27-11eb-9630-d9d960265c76.png)


## Hyperparameter Tuning
I have used logistic regression to predict the target variable, logistic regression is easier to implement, interpret, and very efficient to train. I have used following parameters -
* Regularization parameter C, range used [0.01,0.1,1]
* Maximum number of iterations max_iter, range used [50, 100, 150, 200]

I have performed random sampling over the hyperparameter search space using RandomParameterSampling in our parameter sampler, this drastically reduces computation time and we are still able to find reasonably good models when compared to GridParameterSampling methodology where all the possible values from the search space are used, and it supports early termination of low-performance runs

BanditPolicy is used here which is an "aggressive" early stopping policy. It cuts more runs than a conservative policy like the MedianStoppingPolicy, hence saving the computational time significantly. Configuration Parameters:-

* slack_factor/slack_amount : (factor)The slack allowed with respect to the best performing training run.(amount) Specifies the allowable slack as an absolute amount, instead of a ratio. Set to 0.1.

* evaluation_interval : (optional) The frequency for applying the policy. Set to 1.

* delay_evaluation : (optional) Delays the first policy evaluation for a specified number of intervals. Set to 5.

### Results
The best model has an accuracy of 78.3%. Hyperparameters used are - 

['--C', '1', '--max_iter', '150']


Following screenshots show the successful run of Hyperdrive experiment

![0](https://user-images.githubusercontent.com/6285945/107850393-ff3e8780-6e27-11eb-8a73-cc8134050bf9.png)

![1](https://user-images.githubusercontent.com/6285945/107850394-01084b00-6e28-11eb-91e4-820c842ca0f8.png)

![2](https://user-images.githubusercontent.com/6285945/107850395-02d20e80-6e28-11eb-9734-c594ec8b4a0d.png)

Below are screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![3](https://user-images.githubusercontent.com/6285945/107850397-05ccff00-6e28-11eb-9a03-04a5e0f3a8d9.png)

![4](https://user-images.githubusercontent.com/6285945/107850398-0796c280-6e28-11eb-8a5c-9400c60d14e3.png)

![6](https://user-images.githubusercontent.com/6285945/107850401-0bc2e000-6e28-11eb-96cb-a5391425a33f.png)

![7](https://user-images.githubusercontent.com/6285945/107850403-0e253a00-6e28-11eb-927b-72a53a7f7d46.png)

Saved and registered the best model -

![8](https://user-images.githubusercontent.com/6285945/107850405-0feefd80-6e28-11eb-8fe3-99e02505911e.png)

## Model Deployment
The best model from Hyperdrive experiment has accuracy: 78.3%, whereas the best model from Auto ML experiment has accuracy: 86.9%. So, I have deployed model from Auto ML experiment. The below screen shows the model has been deployed and in Healthy status:

![2](https://user-images.githubusercontent.com/6285945/107850301-6a3b8e80-6e27-11eb-9089-2921431ec4cb.png)

![4](https://user-images.githubusercontent.com/6285945/107850303-6b6cbb80-6e27-11eb-9e25-a4f6f5b62ec5.png)

![5](https://user-images.githubusercontent.com/6285945/107850305-6dcf1580-6e27-11eb-9ef8-af689b11be49.png)

![6](https://user-images.githubusercontent.com/6285945/107850307-6e67ac00-6e27-11eb-8e50-c27fd65f0cd1.png)

![3](https://user-images.githubusercontent.com/6285945/107850302-6ad42500-6e27-11eb-8e56-b77837e1a6cd.png)


For querying the endpoint, one can initiate REST endpoint call using following steps:

* Store scoring uri and primary key
* Create header with key "Content-Type" and value "application/json" and set Authorization with Bearer token
* Create sample input and post the request. Here is a sample input:

`

data= { "data":
       [
           
           {
               'age': 74,
               'anaemia': 1,
               'creatinine_phosphokinase': 1618,
               'diabetes': 1,
               'ejection_fraction': 27,
               'high_blood_pressure': 1,
               'platelets': 275095,
               'serum_creatinine': 2.3,
               'serum_sodium': 133,
               'sex': 0,
               'smoking': 0,
               'time': 9
           },
           {
               'age': 46,
               'anaemia': 0,
               'creatinine_phosphokinase': 800,
               'diabetes': 0,
               'ejection_fraction': 48,
               'high_blood_pressure': 1,
               'platelets': 259000,
               'serum_creatinine': 1.79,
               'serum_sodium': 135,
               'sex': 1,
               'smoking': 0,
               'time': 107
           }
       ]
    }
    
`
    
The below screenshot shows the REST call made to the service and its response:

![7](https://user-images.githubusercontent.com/6285945/107850309-6f98d900-6e27-11eb-962e-a537f3cb05c2.png)

## Screen Recording
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

Please access the screencast using this [link](https://youtu.be/TiGucIh93RM) 


## Standout Suggestions
I have converted the best automl model into ONNX format

![ONNX](https://user-images.githubusercontent.com/6285945/107851638-788ea800-6e31-11eb-9361-64369fa6e4df.JPG)

