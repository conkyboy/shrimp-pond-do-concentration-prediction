# shrimp-pond-do-concentration-prediction
A dissolved oxygen levels prediction method is based on a single-hidden layer feedforward neural network using neighborhood information metrics.

# Abstract
Fluctuations in dissolved oxygen levels directly impact prawn growth and development, serving as a crucial indicator for
monitoring water quality in shrimp ponds. Analyzing and predicting dissolved oxygen content changes due to the intricate
interplay of multiple feature parameters is challenging. Current prediction models for dissolved oxygen must adequately
consider the interactions among feature parameters during data processing to enhance the accuracy of dissolved oxygen
predictions. We propose a technique using neighborhood information entropy to measure the uncertainty of correlation,
redundancy, and interaction between factors and dissolved oxygen. A prediction model (FSNRS-SFNN) has been developed
based on a single-hidden layer feedforward neural network, utilizing neighborhood information metrics to forecast dissolved
oxygen levels in shrimp ponds and optimize shrimp culture practices. %Compared with the multiple linear and ridge regression
models, the FSNRS-SFNN model has the smallest RMSE, MSE, and MAE errors and the highest coefficient of determination.
Compared with multiple linear and ridge regression models, the FSNRS-SFNN model exhibits the most minor mean squared error
(MSE) and mean absolute error (MAE) and the highest coefficient of determination($R^2$). The predictive impact of dissolved
oxygen on shrimp pond management is significant. Therefore, the FSNRS-SFNN model holds excellent potential for
widespread application in predicting dissolved oxygen levels in shrimp ponds.

# Usage
1. Modify the read-in data in the three `.py` files to `DOC-Prediction-DS.csv,` respectively.
2. Execute `Feature Interaction_NCMl.py` to obtain the optimal subset of features.
3. Execute `DNN.py` to obtain the prediction of dissolved oxygen concentration.
3. Execute `RegressionModels.py` to obtain the results of the comparison experiment.
