# Fairness Audit of a Credit Scoring Model Using Fairlearn 

## Project Overview

This project evaluates fairness in a machine learning credit-scoring system using the German Credit dataset. The objective is to determine whether the model produces unequal outcomes across different age groups and to assess how fairness mitigation techniques affect both fairness and predictive performance.
The project was completed as part of a Responsible AI and AI Governance learning portfolio focused on fairness evaluation, model risk assessment, and ethical AI practices.

## Why this matters 
Credit Scoring System are considered to have a high impact on AI application because their decision can highly influence  in access to loan and financial opportunities.
While the machine learning model may get accurate prediction but the doesn't mean their accuracy in demographic groups, outcomes across these groups can still vary.Hence Fairness evaluation is important part of Responsible AI and for model risk management.

## Dataset
 Here we are using german credit dataset from OpenML, 
 this dataset includes customer demographic and financial information such as:Credit history,Loan duration,Savings status,Employment information,Housing information,Age,Existing credits.

## Machine Learning Model
 A Logistic Regression classifier was used as the baseline model.
 Logistic Regression is commonly used for binary classification problems and is considered relatively interpretable compared to more complex machine learning models. This makes it suitable for auditing and governance-focused analysis.   The model was trained to predict whether a customer represented a good or bad credit risk.

## Sensitive Attribute
So, Fairness analysis is done using the age as the sensitive attribute here. We have kept a condition here which is Under 25 years old and 25 years old and above, the main idea behind this project is to determine if the model gives different outcome for the age group. Age was selected as the sensitive attribute because age-related disparities are commonly examined in fairness research and financial decision-making systems.

## Fairness Evaluation
 Fairness was measured using Demographic Parity Difference.This metric compares positive prediction rates across demographic groups.A value closer to 0 indicates more equitable treatment across groups, while larger values indicate greater disparity.

## Bias Mitigation 
Results

Metric	                          |  Before Mitigation	| After Mitigation |
----------------------------------|---------------------|------------------|
 Accuracy	                      |     0.700	        |     0.720        |
 Demographic Parity Difference	  |      0.066	        |     0.065        |

 Fairlearn's ExponentiatedGradient algorithm was used with Demographic Parity constraints to reduce demographic disparity while maintaining model performance.The purpose of mitigation was to improve fairness outcomes without significantly affecting predictive capability.

## Interpretation
The baseline model exhibited relatively low disparity between younger and older applicants, with a Demographic Parity Difference of 0.066.After applying Fairlearn's ExponentiatedGradient mitigation algorithm, the disparity decreased slightly to 0.065 while model accuracy improved from 70.0% to 72.0%.These results suggest that the original model was already relatively balanced across age groups and that fairness mitigation had a limited impact because little demographic disparity was present.The project demonstrates the importance of evaluating fairness directly rather than relying solely on predictive performance metrics.

## Key Governance Insight
- High predictive accuracy does not necessarily imply fair outcomes.
- Fairness should be evaluated separately from model performance.
- Demographic disparities can be measured using fairness metrics.
- Bias mitigation techniques can be applied to reduce identified disparities.
- Responsible AI systems require ongoing monitoring, evaluation, and governance.
## Technologies Used
- Python
- Pandas
- Scikit-learn
- Fairlearn
- Jupyter Notebook

## Future Improvements
- Evaluate additional fairness metrics such as Equalized Odds.
- Compare multiple fairness mitigation approaches.
- Incorporate explainability methods such as SHAP.
- Develop automated fairness monitoring reports.

## Author
Harina M
