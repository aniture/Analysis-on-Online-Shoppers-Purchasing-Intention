# Online Shoppers Purchasing Intention Analysis

This project focuses on analyzing and predicting purchasing intentions of online shoppers using machine learning and data analysis techniques. The dataset was sourced from the UCI Machine Learning Repository, and the goal was to identify key factors influencing purchasing behavior and develop predictive models for actionable insights.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Technologies Used](#technologies-used)
4. [Workflow](#workflow)
5. [Methods and Models](#methods-and-models)
6. [Results and Insights](#results-and-insights)
7. [Future Work](#future-work)
8. [How to Run](#how-to-run)
9. [Contributors](#contributors)

---

## Overview
The primary objective of this project was to explore the relationships between various website metrics and purchasing behavior. We utilized a combination of exploratory data analysis (EDA), feature engineering, and machine learning techniques to predict whether a customer would make a purchase.

---

## Dataset Description
- **Source**: UCI Machine Learning Repository.
- **Size**: 12,330 sessions.
- **Features**:
  - Numerical: Administrative Duration, Informational Duration, ProductRelated Duration, etc.
  - Categorical: Month, VisitorType, Weekend, etc.
- **Target Variable**: `Revenue` (True for purchase, False otherwise).

---

## Technologies Used
- **Programming Languages**: Python
- **Libraries**:
  - Data Analysis: `Pandas`, `NumPy`
  - Visualization: `Matplotlib`, `Seaborn`
  - Machine Learning: `scikit-learn`, `XGBoost`, `imbalanced-learn`
  - Dimensionality Reduction: `UMAP`, `t-SNE`
- **Tools**: Jupyter Notebook

---

## Workflow
1. **Data Preprocessing**:
   - Handled missing values and categorical encoding.
   - Standardized numerical features and balanced the dataset using SMOTE.
2. **Exploratory Data Analysis (EDA)**:
   - Explored correlations between features and purchasing behavior.
   - Identified key patterns and trends in customer activity.
3. **Dimensionality Reduction**:
   - Applied PCA, UMAP, and t-SNE for feature visualization.
4. **Modeling**:
   - Supervised Learning: Naive Bayes, KNN, Random Forest, SVM, XGBoost.
   - Unsupervised Learning: K-Means, DBSCAN, Hierarchical Clustering.
5. **Evaluation**:
   - Measured model performance using metrics like accuracy, AUC, and F1-score.

---

## Methods and Models
### Dimensionality Reduction
- **PCA**: Highlighted the most significant features and reduced the complexity for visualization.
- **UMAP**: Effectively separated clusters of purchasing and non-purchasing customers.
- **t-SNE**: Provided detailed visualizations of local data relationships.

### Supervised Models
- **Naive Bayes**: Baseline performance with moderate accuracy.
- **Random Forest**: Achieved high accuracy and identified top features influencing purchasing behavior.
- **XGBoost**: Delivered the best results with an AUC of 0.93 using early stopping and hyperparameter tuning.

### Unsupervised Models
- **K-Means**: Identified clusters that grouped customers with similar behaviors.
- **DBSCAN**: Detected patterns in noisy data, but required careful tuning.
- **Hierarchical Clustering**: Visualized relationships between different customer segments.

---

## Results and Insights
1. Customers spending more time on product-related pages are more likely to purchase.
2. Lower bounce rates and higher page values correlate strongly with purchasing behavior.
3. Random Forest and XGBoost models outperformed others in predicting purchases, with accuracies of 89% and 90%, respectively.
4. Dimensionality reduction techniques like UMAP and t-SNE effectively visualized data clusters.

---

## Future Work
1. Implement Gaussian Mixture Models for clustering.
2. Explore additional datasets for similar analysis.
3. Develop end-to-end pipelines using MLOps practices.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/online-shoppers-intention-analysis.git
   cd online-shoppers-intention-analysis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the project notebook and follow the workflow.

---

## Contributors
- **Shaktisharan Rajure**: Data preprocessing, modeling, and evaluation.
- **Aditya Niture**: Exploratory data analysis, visualizations, and insights.

---

Feel free to customize this content further to fit your repository or project needs. Let me know if you need assistance setting up the repository or requirements file!