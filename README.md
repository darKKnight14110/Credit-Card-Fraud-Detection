It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The main objective of the analysis is to classify the heavily unbalanced dataset using many different classification methods.The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Through the analysis , we understand which classification method is the best for the current dataset and can be used for future classification on test-data . Since , we have a heavily unbalanced data , we choose the metric for scoring as the "F1-Score" .Classification models used are:
1.	Logistic Regression
2.	Random Forest Classifier(with GridSearchCV)
3.	SVC Classifier
4.	XBG Classifier(with GridSearchCV)
5.	MLP Classifier
6.	Stacking Classifier
