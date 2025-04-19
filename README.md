# Exploring-Machine-Learning-Applications-in-Predicting-Heart-Disease-risk
A Comparative Study of Logistic Regression, SVM,  and Decision Trees

Problem Statement & Purpose

Heart disease is one of the leading causes of death globally, yet early diagnosis remains a major challenge in clinical settings. Traditional risk calculators often fail to identify high-risk individuals in time for preventative interventions. In response to this, the healthcare industry is increasingly turning to artificial intelligence (AI) and machine learning (ML) to enhance diagnostic accuracy and decision-making.

The purpose of this project is to investigate how machine learning can be applied to predict heart disease risk using structured clinical data. By evaluating and comparing three widely used ML models Logistic Regression, support Vector Machine (SVM), and Decision Trees. This research aims to determine which algorithm provides the most reliable predictions and is most suitable for clinical integration.

What This Project Covers

This project uses the Heart Disease UCI dataset from Kaggle to develop and evaluate three supervised ML models. The models are assessed based on five key performance metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

In addition to model training and testing, the project includes:
- Preprocessing and normalization of clinical features
- Visual analysis using confusion matrices and ROC curves
- Performance comparison against peer-reviewed literature
- Discussion of real-world applicability in healthcare settings

Dataset

Heart Disease UCI Dataset 
Available on Kaggle: [https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci]

The dataset contains 303 records with 14 clinical attributes including:
- Age
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Chest pain type
- Resting ECG results
- Max heart rate
- Exercise-induced angina
- ST depression, slope, and other key indicators

The target variable (`condition`) indicates presence (1) or absence (0) of heart disease.

Tools and Methodology

- Python (Google Colab)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Preprocessing:
- Missing values handled
- Features scaled using `StandardScaler` for Logistic Regression and SVM
- Dataset split into training and test sets (80/20 ratio)

Models Implemented:
- **Logistic Regression** — a linear model for binary classification
- **Support Vector Machine (SVM)** — a margin-based classifier
- **Decision Tree** — a tree-based model with rule-based splitting

Evaluation Metrics:
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Model Results

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|----------|
| Logistic Regression| 0.733    | 0.700     | 0.750  | 0.724    | 0.842    |
| SVM                | 0.733    | 0.688     | 0.786  | 0.733    | 0.834    |
| Decision Tree      | 0.767    | 0.733     | 0.786  | 0.759    | 0.768    |

- Decision Tree performed best overall with highest accuracy and F1 score.
- Logistic Regression and SVM had stronger AUC scores, showing good class separation.

Visualizations

- Confusion matrices for each model to show prediction breakdown
- ROC curves with AUC values for visual model comparison

Visuals are available in the file: `model_visuals.png`

Literature Support

This project was informed by peer-reviewed research demonstrating the effectiveness of ML in cardiovascular risk prediction:

- Weng et al. (2017)  Showed ML models outperform traditional risk calculators.
- Deo (2015) Highlighted ML’s role in reducing diagnostic errors.
- Gudadhe et al. (2010) Demonstrated strong performance of SVM and neural networks.
- Alizadehsani et al. (2018)– Compared ML methods for coronary artery diagnosis.
- IEEE CONECCT (2022) – Confirmed the usefulness of comparing multiple supervised ML algorithms.
- Detrano et al. (1989)– Provided clinical validation for the UCI dataset structure.

Project Status

Model selection and training complete  
Metrics evaluated and interpreted  
PowerPoint presentation and documentation finalized  
Visualizations generated  
Deployed on GitHub for open-source access and future learning

This project is complete and was submitted as part of the SAT5114 AI in Healthcare course requirements.

Citation Examples

- Weng, S. F., Reps, J., Kai, J., et al. (2017). PLoS ONE, 12(4), e0174944.
- Gudadhe, M., Wankhade, K., & Dongre, S. (2010). IEEE ICCCT.
- Heart Disease Dataset – Kaggle: [link](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

Author

Asiana Holloway 
Graduate Student – AI in Healthcare  
GitHub: AsianaHolloway

