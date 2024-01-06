It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The main objective of the analysis is to classify the heavily unbalanced dataset using many different classification methods. Through the analysis , we understand which classification method is the best for the current dataset and can be used for future classification on test-data . Since , we have a heavily unbalanced data , we choose the metric for scoring as the “AUC” .Classification models used are 
1.	Logistic Regression 
2.	Logistic Regression (where weights are assigned to classes)
3.	K- Nearest Neighbours
4.	Decision Tree Classifier ( with GridSearchCV)
5.	Bagging Classifier ( with base_estimator as DecisionTreeClassifier )
6.	Adaboost Classifier ( with GridSearchCV)
7.	Stacking Classifier ( with GradientBoostingClassifier & LogisticRegressor as estimators and XGB as the final estimator) 
The best scorer was Logistic Regressor 
