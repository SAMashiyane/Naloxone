<div align="center">
<h1 align="center"><strong>🛣  Prediction of naloxone dose in opioids toxicity based on machine learning techniques (artificial intelligence)</strong></h1>

  
 ![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
 ![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)]
 ![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.svg?style=for-the-badge)]
 ![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)
 
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
- [Acknowledgements](#acknowledgements)

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

- <b> prerequisite for training</b>:
The Anaconda Distribution, commonly known as Anaconda, is one of the most renowned Python distribution platforms.
It is a popular tool for data science and machine learning developers. This is because it offers a collection of over 800 packages installed and curated to work correctly out of the box.The Anaconda distribution is also free and very user-friendly. It comes with a command-line interface for terminal nerds and the Anaconda navigator, allowing you to manage environments and packages with a GUI interface.
While installing packages in requirements.txt using Conda through the following command:
```shell
 conda install --yes --file requirements.txt
```

### 💻 Setup <a name="setup"></a>







