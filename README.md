# Supervised Machine Learning

## Case Study
Please refer to the [Report](report.md) file and Jupyter notebook [credit_risk_resampling.ipynb](credit_risk_resampling.ipynb)
file.

## Technical Environment
This tool utilises the following technologies:
- **Pandas** [Documentation](https://pandas.pydata.org/docs/reference/frame.html)
- **NumPy** [Documentation](https://numpy.org/)
- **Scikit learn** [Documentation](https://scikit-learn.org/stable/)
- **Imbalanced learn** [Documentation](https://imbalanced-learn.org/stable/)

## Glossary
**Accuracy** is a measure of how often the model is correct. The ratio of correctly predicted observations to the total 
number of observations. It doesn't always communicate how precise the model is. Accuracy can be very susceptible to 
imbalanced classes.
```
(TP + TN) / (TP + TN + FP + FN)
```
**Precision** is the ratio of correctly predicted positive observations to the total predicted positive observations. 
High precision relates to a low false positive rate.
```
TP / (TP + FP)
```
**Recall** is the ratio of correctly predicted positive observations to all predicted observations for that class.
```
TP / (TP + FN)
```
**Classification Report** identifies the Precision, Recall and Accuracy of a model for each given class.
**Confusion Matrix**

Source: Monash University, [FinTech Boot Camp](https://bootcamps.monash.edu/fintech) learning material.
