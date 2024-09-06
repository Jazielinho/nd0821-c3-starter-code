# Model Card

## Model Details

**Model type**: Binary classification model predicting whether an individual's salary is above or below $50K/year.  
**Algorithm**: The model is a supervised learning classifier (e.g., logistic regression, decision tree, etc.).  
**Inputs**: Categorical and numerical features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.  
**Outputs**: Predicted class label (0 or 1) representing salary range.  

## Intended Use

This model is designed to predict income levels based on various demographic and work-related features. It is intended for use in social studies, demographic analysis, and to aid in resource allocation, providing insights into salary inequality patterns based on race, gender, and other factors.

The model is **not** intended for:

- Making definitive conclusions about individuals' salaries.
- Use in high-stakes decision-making such as hiring or financial assessments without appropriate fairness audits and evaluations.


## Training Data

The model was trained on the UCI Adult Census dataset. The training data was split into 80% training and 20% testing sets. Categorical features include workclass, education, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data

Evaluation was done using a test set (20% of the data) that was not seen by the model during training. The same preprocessing steps were applied to the test data, including encoding categorical features and scaling numerical ones.

## Metrics
The following metrics were used to evaluate the model:

- **Precision**: Measures the proportion of positive identifications (salary >50K) that were actually correct.
- **Recall**: Measures the proportion of actual positives that were identified correctly.
- **F-beta score**: A weighted harmonic mean of precision and recall, with beta determining the weight of recall.


### Overall Metrics (Test Set)
- Precision: 0.7057
- Recall: 0.6243
- F-beta score: 0.6625

### Metrics by Race

| Race                | Precision | Recall  | F-beta |
|---------------------|-----------|---------|--------|
| White               | 0.7056    | 0.6309  | 0.6662 |
| Black               | 0.7692    | 0.5263  | 0.6250 |
| Asian-Pac-Islander  | 0.6316    | 0.6102  | 0.6207 |
| Amer-Indian-Eskimo  | 1.0000    | 0.5000  | 0.6667 |
| Other               | 0.6667    | 0.6667  | 0.6667 |


### Metrics by Sex:

| Sex    | Precision | Recall  | F-beta |
|--------|-----------|---------|--------|
| Male   | 0.7040    | 0.6399  | 0.6704 |
| Female | 0.7176    | 0.5351  | 0.6131 |


### Training Metrics:

- Precision: 1.0000
- Recall: 0.9998
- F-beta score: 0.9999

## Ethical Considerations

- **Bias**: The model may exhibit bias across race and gender, as reflected in varying metrics for different demographic groups. Precision and recall differ significantly across racial categories and between sexes, highlighting potential inequalities in predictions.
- **Fairness**: The model should be carefully examined for fairness, especially in applications where decisions may affect individuals based on income predictions.
- **Transparency**: Users of the model should be aware of the dataset's limitations, particularly the historical biases in income and demographics present in the U.S. Census data.


## Caveats and Recommendations

- **Data Quality**: The model's performance is heavily dependent on the quality and representation of the data. Any biases or errors in the input data will be reflected in the model's predictions.
- **Generalization**: This model was trained on the UCI Adult Census dataset, which may not generalize well to other populations or datasets without additional tuning and validation.
- **Bias Mitigation**: Regular audits should be conducted to check for biases across race, gender, and other protected categories.