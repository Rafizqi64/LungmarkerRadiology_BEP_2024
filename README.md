
# Minimally Invasive Clinical Prediction Model for Small Pulmonary Nodules in Non Small Cell Lung Cancer Patients

## Project Overview
This research evaluates the performance
of ensemble models combining radiology based Brock and Herder models with liquid biopsy based models
for the classification of NSCLC from pulmonary nodules. The models include logistic regression, decision trees, and ensemble methods that aim to give a risk assesment of NSCLC and treatment outcomes.

## Features
- Data preprocessing and cleaning.
- Exploratory data analysis
- Implementation of multiple predictive models.
- Model metric display
- Evaluation of models using statistical tests, ROC curves, prediction histograms and confusion matrices.
- Feature importance analysis using SHAP values.

## Installation

### Prerequisites
- Python 3.6+
- Pip package manager


### Setup
Clone this repository to your local machine:
```bash
git clone https://github.com/Rafizqi64/LungmarkerRadiology_BEP_2024.git
cd LungmarkerRadiology_BEP_2024 # Just make sure you are in the right directory 
```

Install the required packages:
```bash
pip install -r requirements.txt
```
Copy your .xlsx dataset into the same directory as the python scripts or adjust the filepath in the code to the location of the dataset.

## Usage

### main.py
'main.py' contains most of the evaluation and training procedures and acts as easy access to running and plotting your chosen models with a range of adjustments.
```bash
python main.py
```
The script is divided into several sections, each handling a specific part of the machine learning pipeline:

1. **Data Loading and Preprocessing**:
   - Loads data from an Excel file. ***MAKE SURE YOU ADJUST THIS***
   - Applies transformations and preprocessing such as one-hot encoding, log transformations, and scaling.

2. **Feature Definition**:
   - Specifies the features for different models based on clinical indicators and imaging results.
   - Allows easy customization of features used in the models.

3. **Model Manager Initialization and Training**:
   - Adds different predictive models.
   - Configures models with specified features.
   - Trains models using cross-validation, 200x crossvalidation or SMOTE with PPV and NPV thresholds and evaluates their performance.
   - Training is done in 1 go and is parsed to the voting classifier input models

4. **Feature Selection**:
   - Applies various feature selection techniques to refine the model inputs.
   - Techniques include tree-based selection, recursive feature elimination, and L1 regularization.

5. **Statistical Tests**:
   - Performs statistical comparisons between model predictions and guideline-based outcomes using tests such as the Mann-Whitney U test and Wilcoxon signed-rank test.

6. **Retrained Models**:
   - Generates ROC curves, SHAP value plots, prediction histograms, and confusion matrices for the retrained models.
   - Also prints the metrics and std.

7. **Hybrid Models**:
   - Plots the logistic regression ensemble models combining predictions from features and original model scores to improve performance.
   - Uses the model manager to train the models
   - Evaluates ensemble models in the same detailed manner as individual models.
     
8. **Input Models**:
   - Constructs voting classifier ensemble models to combine predictions from all retrained models added to the model manager.
   - Uses the voting classifier class in 'ensemble_model.py' to train the model
   - Evaluates ensemble models in the same detailed manner as individual models.

9. **Output Models**:
   - Plots the logistic regression ensemble models combining predictions from only model scores.
   - Uses the model manager to train the models
   - Evaluates ensemble models in the same detailed manner as individual models.
   
Simply comment and uncomment the lines of code to specify which model you want trained and evaluated.

### EDA.py

The script is organized into a class `ModelEDA` that encapsulates all EDA functionalities and evaluation of original model metrics.  
The structure is as follows:

1. **Initialization and Data Loading**:
   - Instantiates the class with a path to the dataset. ***MAKE SURE YOU ADJUST THIS***
   - Loads the dataset and initializes variables for analysis.

2. **Distribution Analysis**:
   - Calculates and prints the distribution of both categorical and numerical variables within the dataset.

3. **Significance Testing**:
   - Performs statistical tests such as Fisher's Exact Test, Chi-Square, and logistic regression to explore associations between features and the target variable.
   - Evaluates the significance of the differences in model scores across categorical features.

4. **Visualization**:
   - Generates various plots including histograms, ROC curves, and scatter plots to visualize data distributions, relationships between variables, and model performances.
   - Plots confusion matrices to visualize model prediction accuracies at different thresholds.

5. **Model Score Evaluation**:
   - Evaluates original model scores using metrics such as accuracy, precision, recall, F1-score, and ROC AUC using cross-validation.
Uncomment the functions you want to run and run the file.

### decisiontree.py

The script is organized into a class `DecisionTree` that encapsulates functionalities of evaluating and parsing BTS Guideline data for a statistical comparison against the ensemble models. It is built out of:

1. **Initialization and Data Loading**:
   - Instantiates the class with a path to the dataset.
   - Loads the dataset and initializes variables for analysis.

2. **Guideline Logic Application**:
   - Applies BTS guideline logic to predict outcomes based on model scores and other clinical features.

3. **Metrics Evaluation**:
   - Uses Stratified K-Fold cross-validation to evaluate the accuracy, precision, recall, F1-score, and specificity of the guideline predictions.

4. **ROC Curve Plotting**:
   - Generates ROC curves to assess the performance of the guideline-based predictions.

5. **Confusion Matrix Plotting**:
   - Visualizes confusion matrices to understand the true positive, false positive, true negative, and false negative rates.
Uncomment the functions you want to run and run the file.

### data_preprocessing.py, model.py, herder_model.py and ensemble_model.py
These scripts are called upon in 'main.py' for preprocessing data, training and evaluating models, and plotting graphs.
Documentation is given in the code itself and is hopefully sufficient.

## Models Included
***Retrained Models*** (Utilizing feature selection on the original features)
- ***Brock model (Logistic Regression)***
- ***Herder model (Logistic Regression):*** 2 step training with original features
- ***Herbert model (Logistic Regression)*** 1 step training with original features and No FDG Avidity
- ***LC model (Logistic Regression)***
- ***NSCLC model (Logistic Regression)***
- ***LBx model (Logistic Regression):*** Liquid biopsy model using feature selected markers

***Output Models:*** (Utilizing original model scores)}
- ***LBx + Brock and Herder output (Soft Voting Classifier):*** Averages the original LC and NSCLC scores with Brock and Herder scores.
- ***Brock and Herder output (Soft Voting Classifier):*** Averages solely the original Brock and Herder scores.
- ***Global feature Model:*** (Utilizing all features from each model together)
- ***LBx + Brock and Herder features (Logistic Regression):*** Incorporates all features selected by feature selection from LBx, Brock, and Herder.

***Input Models: (Utilizing the input features from each model)***
- ***LBx + Brock and Herder input (Soft Voting Classifier):*** Averages predictions of the retrained LBx, Brock, and Herder models based on selected features.
- ***Brock and Herder input (Soft Voting Classifier):*** Averages predictions of the retrained Brock and Herder models based on selected features.

***Hybrid Models (Utilizing both features and original model output):***
- ***NSCLC output + Brock and Herder features (Logistic Regression):*** Combines original NSCLC output with features from Brock and Herder.
- ***LC output + Brock and Herder features (Logistic Regression):*** Combines original LC output with features from Brock and Herder.
- ***LBx features + Brock and Herder output (Logistic Regression):*** Merges tree based selected protein marker features with the original Brock and Herder scores.


## Authors
- **Rafi van Kruchten** - [Rafizqi64](https://github.com/Rafizqi64) - [Email](r.r.f.v.kruchten@student.tue.nl)
