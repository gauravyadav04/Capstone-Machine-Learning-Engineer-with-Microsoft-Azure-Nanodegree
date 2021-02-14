# Predict mortality caused by heart failure

This project is part of the Udacity Azure ML Nanodegree. In this project, I have used [heart failure clinical records dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). 

The objective is to train a machine learning model using Hyperdrive and AutoML and predict mortality caused by heart failure. Models from these two experiments are compared using Accuracy score, best model is registered and deployed to Azure Container Service as a REST endpoint with key based authentication.


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
The following screenshots show the successfully completed AutoML run. The best model from this experiment was the VotingEnsemble with an `Accuracy` of 87.3%

![1](https://user-images.githubusercontent.com/6285945/107873645-f1493f00-6ed9-11eb-8cb4-dbfc3751a1e0.png)

![2](https://user-images.githubusercontent.com/6285945/107873647-f5755c80-6ed9-11eb-84f7-3dec987b8973.png)

![3](https://user-images.githubusercontent.com/6285945/107873649-f8704d00-6ed9-11eb-98b9-a05b34cd8296.png)

Below are the screenshots of `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![4](https://user-images.githubusercontent.com/6285945/107873651-fad2a700-6ed9-11eb-9b10-121e8e1e40f4.png)

![5](https://user-images.githubusercontent.com/6285945/107873654-fdcd9780-6ed9-11eb-8b0b-545628d6b76c.png)

![6](https://user-images.githubusercontent.com/6285945/107873657-00c88800-6eda-11eb-8e6e-25998d7f6dbe.png)

![7](https://user-images.githubusercontent.com/6285945/107873658-032ae200-6eda-11eb-85b4-d9092fa82706.png)

VotingEnsemble takes a majority vote of various algorithms, this make it extremely robust and helps reduce the bias associated with individual estimators. Following are the estimators and their respective weights used in VotingEnsemble trained in Auto ML experiment

| Estimator | Weight |
| ------ | ------ |
| extratreesclassifier with robustscaler | 0.14285714285714285 |
| extratreesclassifier with maxabsscaler | 0.14285714285714285 |
| randomforestclassifier with robustscaler | 0.14285714285714285 |
| xgboostclassifier with sparsenormalizer | 0.14285714285714285 |
| lightgbmclassifier with sparsenormalizer | 0.14285714285714285 |
| xgboostclassifier with standardscalerwrapper | 0.2857142857142857 |

Below are the details of all parameters associated with estimators used in VotingEnsemble

```
datatransformer
{'enable_dnn': None,
 'enable_feature_sweeping': None,
 'feature_sweeping_config': None,
 'feature_sweeping_timeout': None,
 'featurization_config': None,
 'force_text_dnn': None,
 'is_cross_validation': None,
 'is_onnx_compatible': None,
 'logger': None,
 'observer': None,
 'task': None,
 'working_dir': None}

prefittedsoftvotingclassifier
{'estimators': ['14', '30', '28', '11', '34', '32'],
 'weights': [0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.2857142857142857]}

14 - robustscaler
{'copy': True,
 'quantile_range': [25, 75],
 'with_centering': True,
 'with_scaling': False}

14 - extratreesclassifier
{'bootstrap': False,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'entropy',
 'max_depth': None,
 'max_features': 0.8,
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.01,
 'min_samples_split': 0.2442105263157895,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 100,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

30 - maxabsscaler
{'copy': True}

30 - extratreesclassifier
{'bootstrap': False,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'entropy',
 'max_depth': None,
 'max_features': 0.9,
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.08736842105263157,
 'min_samples_split': 0.19736842105263158,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 25,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

28 - robustscaler
{'copy': True,
 'quantile_range': [10, 90],
 'with_centering': True,
 'with_scaling': False}

28 - randomforestclassifier
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 0.4,
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.035789473684210524,
 'min_samples_split': 0.10368421052631578,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 25,
 'n_jobs': 1,
 'oob_score': True,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}

11 - sparsenormalizer
{'copy': True, 'norm': 'max'}

11 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.9,
 'eta': 0.3,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 10,
 'max_leaves': 15,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 25,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 0.5208333333333334,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.6,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

34 - sparsenormalizer
{'copy': True, 'norm': 'l1'}

34 - lightgbmclassifier
{'boosting_type': 'goss',
 'class_weight': None,
 'colsample_bytree': 0.4955555555555555,
 'importance_type': 'split',
 'learning_rate': 0.05263631578947369,
 'max_bin': 10,
 'max_depth': 10,
 'min_child_samples': 9,
 'min_child_weight': 2,
 'min_split_gain': 1,
 'n_estimators': 50,
 'n_jobs': 1,
 'num_leaves': 110,
 'objective': None,
 'random_state': None,
 'reg_alpha': 0.5789473684210527,
 'reg_lambda': 0.10526315789473684,
 'silent': True,
 'subsample': 1,
 'subsample_for_bin': 200000,
 'subsample_freq': 0,
 'verbose': -10}

32 - standardscalerwrapper
{'class_name': 'StandardScaler',
 'copy': True,
 'module_name': 'sklearn.preprocessing._data',
 'with_mean': False,
 'with_std': False}

32 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.6,
 'eta': 0.4,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 7,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 10,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 0.20833333333333334,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.6,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}
```

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
The best model from Hyperdrive experiment has accuracy: 78.3%, whereas the best model from Auto ML experiment has accuracy: 87.3%. I registered the model from Auto ML experiment, then created InferenceConfig by providing the entry script [score.py](https://github.com/gauravyadav04/Capstone-Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree/blob/main/score.py) and [environment dependencies](https://github.com/gauravyadav04/Capstone-Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree/blob/main/my-conda-env.yml). After that I deployed model as a web service with ACI (Azure Container Instance) using deploy configuration -

* cpu_cores = 1,
* memory_gb = 1,
* auth_enabled = True,
* enable_app_insights = True,
* tags = {'name':'Heart failure prediction'},
* description='Heart failure prediction model'

![2](https://user-images.githubusercontent.com/6285945/107850301-6a3b8e80-6e27-11eb-9089-2921431ec4cb.png)

![4](https://user-images.githubusercontent.com/6285945/107850303-6b6cbb80-6e27-11eb-9e25-a4f6f5b62ec5.png)

![5](https://user-images.githubusercontent.com/6285945/107850305-6dcf1580-6e27-11eb-9ef8-af689b11be49.png)

![6](https://user-images.githubusercontent.com/6285945/107850307-6e67ac00-6e27-11eb-8e50-c27fd65f0cd1.png)

![3](https://user-images.githubusercontent.com/6285945/107850302-6ad42500-6e27-11eb-8e56-b77837e1a6cd.png)


For querying the endpoint, one can initiate REST endpoint call using following steps:

* Store scoring uri and primary key
* Create header with key "Content-Type" and value "application/json" and set Authorization with Bearer token
* Create sample input and post the request. 
    
The below screenshot shows the REST call made with sample input to the service and its response:

![Screenshot (478)](https://user-images.githubusercontent.com/6285945/107873769-e0e59400-6eda-11eb-99c3-87e218e7ed28.png)

## Screen Recording
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

Please access the screencast using this [link](https://youtu.be/TiGucIh93RM) 


## Standout Suggestions
I have converted the best automl model into ONNX format

![ONNX](https://user-images.githubusercontent.com/6285945/107851638-788ea800-6e31-11eb-9361-64369fa6e4df.JPG)

## Further Improvement

* Model performance can be improved by collecting more data
* Consider other performance metrics such as Precision, Recall and F1 Score 
* Deploy model to Edge using Azure IoT Edge

