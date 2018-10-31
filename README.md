# Loan-data-analysis

Experimental analysis of a labeled loan dataset

Summary: 

We analyse labelled loan application data having 614 observations and 11 features applying supervised machine learning algorithms including neural nets. The data has been obtained from Analytics Vidhya.

After some EDA and data munging, we fit the standard classification models from scikitlearn like Logistic Regression, SVC, Linear SVC, 
Random Forest, KNN etc. We introduce some extra features that improve the performance a little bit. Feature importances (or coefficients) 
more or less conform to the EDA. Logistic, Linear SVC and Random Forest are the best performers.

However, all models suffer from a very weak recall score (about 0.45) for the class 0 (loan rejected). Two possible shortcomings may be: 
1) Models not able to produce desired nonlinearlity of decision boundary or 2) Models having diffculty with noisy decison boundary, 
possibly aggravated by unbalanced classes ( 69 % approved versus 31 % rejected).

Since Logistic and Linear SVC have performed better than SVC, the second possiblity is more likely to be playing the spoiler. So we try 
the AdaBoost Classifier, which tries to recitify things by assigning higher weights to misclassified observations. We use Random Forest 
as the base classifer for AdaBoost. This has a good impact on recall score for class 0 (0.55), but precision for this class goes down
(0.75) and f-1 score does not change(0.64). Weighted average of f1 is 0.80, better than our previous best of 0.79.

However, we know that neural networks are much better at discovering nonlinear decision boundaries. So to double-check, we build a neural 
net from scratch (with one hidden layer, He initialization, and the options of using Momentum or Adam optimizations). The net does not 
outperform Logistic Regression. Recall for class 0 is again 0.45.

We do a final experiment: We know that neural nets love bigger datasets. Can we adress the imbalance in the classes by producing a bigger 
and more balanced training set using random sampling with replacement? We carry this out. For class 0 we get recall=0.63, precision=0.65, 
f1=0.64. Weighted average of f1 is again 0.80. So this is as good as AdaBoost, and a bit more balanced even.

The details are available in the jupyter notebook "Loan_pred_ens.ipynb"
