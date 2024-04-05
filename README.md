<div align="center">
<h1 align="center"><strong>  Prediction of naloxone dose in opioids toxicity based on machine learning techniques (artificial intelligence)</strong></h1> 
 
 ![Python - Version]( https://img.shields.io/badge/Python-3.9+-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
 ![scikit_learn - Version](https://img.shields.io/badge/scikit_learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![anaconda -version](https://img.shields.io/badge/conda-4.x-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)
 ![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)
 ![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)
 
 
</div>

----

## 📚 Table of Contents
- [Abstract](#Abstract)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup ](#setup)
  - [Running the project ](#running-the-project)
- [Citing](#citing)
- [License](#license)

---- 
## 📌 Abstract <a name="Abstract"></a>
Treatment management for opioid poisoning is critical and, at the same time, requires specialized knowledge and skills. This study was designed to develop and evaluate machine learning algorithms for predicting the maintenance dose and duration of hospital stay in opioid poisoning, in order to facilitate appropriate clinical decision-making.
This study used artificial intelligence technology to predict the maintenance dose and duration of administration by selecting clinical and paraclinical features that were selected by Pearson correlation (filter method) (Stage 1) and then the (wrapper method) Recursive Feature Elimination Cross-Validated (RFECV) (Stage2).
The duration of administration was divided into two categories: A (which includes a duration of less than or equal to 24 hours of infusion) and B (more than 24 hours of naloxone infusion). XGBoost was found to be the superior model.the most important features for classifying patients for the duration of treatment were bicarbonate, respiration rate, physical sign, The partial pressure of carbon dioxide (PCO2), diastolic blood pressure, pulse rate, naloxone bolus dose, Blood Creatinine(Cr), Body temperature (T)...

----
## 💫 Demo <a name="demo"></a>

![](https://github.com/SAMashiyane/Naloxone/blob/main/figures/RFECV_XGBClassifier.gif)
![](https://github.com/SAMashiyane/Naloxone/blob/main/figures/Feature.gif)

----
## 🚀 Getting Started <a name="getting-started"></a>

### ✅ Prerequisites <a name="prerequisites"></a>

- <b> dependencies</b>:

The Anaconda Distribution, commonly known as Anaconda, is one of the most renowned Python distribution platforms.
It is a popular tool for data science and machine learning developers. This is because it offers a collection of over 800 packages installed and curated to work correctly out of the box.The Anaconda distribution is also free and very user-friendly. It comes with a command-line interface for terminal nerds and the Anaconda navigator, allowing you to manage environments and packages with a GUI interface.
While installing packages in requirements.txt using Conda through the following command:Install the dependencies:
```shell
 conda install --yes --file requirements.txt
```

### 💻 Setup <a name="setup"></a>

1. Clone the repository:
 ```shell
 git clone https://github.com/SAMashiyane/Naloxone.git
 ```
 2. Change to the project directory:
 ```shell
 cd src
 ```
 3. Setting up programming environment to run the project:
 
 - If using an installed <a hre="https://docs.conda.io/en/latest/">conda</a> package manager, i.e. either Anaconda or Miniconda, create the conda environment following the steps mentioned below:
 ```shell
 conda create --name <environment-name> python=3.9.x
 conda activate <environment-name>
 ```

### 🤖 Running the project <a name="running-the-project"></a>
1. naloxlib libraries was created specifically for this project. importing naloxalib .
```shell
import naloxlib
from naloxlib.classifier import * #  ---> for use classification stage 
```
2. build_naloxone_model for classification
```shell
build_naloxone_model(data=data_selection,session_id=123,train_size = 0.7)
```
3. Comparing All Models
```shell
Classifier_comparison_naloxone()
```
4. for use Id_Model or Id_plot :
```shell
help(Id_index)
```
output: Model name:
 
 |    Name_Model                   |          Id_Model           |
 |---------------------------------|-----------------------------|
 |                                 |                             |
 | LogisticRegressionClassifier    |          LogReg             |
 | KNeighborsClassifie             |          KNN                |
 |  GaussianNBClassifie            |          GNB                |
 | DecisionTreeClassifier          |         DT                  |
 | SVM - Linear Kernel             |         SVM                 |
 | Gaussian Process Classifier     |          GauProC            |
 | MLP Classifier                  |           MLP               |
 | Ridge Classifier                |           RIG               |
 | Random Forest Classifier        |           RanForest         |
 | Ada Boost Classifier            |           AdaBo             |
 | Gradient Boosting Classifier    |           GraBoC            |
 | Linear Discriminant Analysis    |           LDisAn            |
 | Extra Trees Classifier          |           EXTre             |
 | Extreme Gradient Boosting       |           xgboost           |
 | Light Gradient Boosting Machine |        lightgbm             |
 | CatBoost Classifier             |        catboost             |
 | Dummy Classifier                |        Dummy                |
 | Calibrated Classifier CV        |        CalibratedCV         |
  
  Plot_machine : this function power from Yellowbrick
  
 |    Id_plot               |          Name Plot               |
 |--------------------------|----------------------------------|
 |                          |                                  |
 |    "auc"                 |              "AUC"               |
 |    "confusion_matrix"    |    "Confusion Matrix"            |
 |    "pr"                  |       "Precision Recall"         |
 |    "error"               |      "Prediction Error"          |
 |    "class_report"        |      "Class Report"              |
 |    "learning"            |       "Learning Curve"           |
 |    "feature"             | "Feature Importance"             |
 | "feature_all"            |  "Feature Importance (All)"      |
 |  "rfe"                   | "Feature Selection"              |
 
               


5. Make Models
```shell
NameML = make_machine_learning_model('Id_Model') 
```
6. Prediction on Test Sample(test_dataset) 
```shell
predict_model(Id_Model);
```
7. Machine learning plot
```shell
plot_machine(Id_Model, plot = 'Id_plot') 
```

----
## 📝 Citing <a name="citing"></a>
```
This section will be dedicated to article citation in the future
```
----

## 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/SAMashiyane/Naloxone/blob/main/LICENSE)

<p align="right">
 <a href="#top"><b>🔝 Return </b></a>
</p>

------





