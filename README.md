# CementStrengthPrediction
The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period. I solved this problem using Data science and Machine learning technology, developed a web application which predicts the "Concrete compressive strength" based on the quantities of raw material, given as an input. Sounds like this saves a lot of time and effort right !

Data source:- https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set

# Approach:
Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.

# Performed Exploratory data analysis:
First viewed the distribution of the target feature, "Concrete compressive strength", which was in Normal distribution with a very little right skewness.
Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.

To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
Checked for the presence of outliers in all the columns and found that the column 'age' is having more no. of outliers. Removed outliers using IQR technique.

# Experimenting with various ML algorithms:
First, tried with Linear regression models and feature selection using Backward elimination, RFE and the LassoCV approaches. Stored the important features found by each model into "relevant_features_by_models.csv" file into the "results" directory. Performance metrics are calculated for all the three approaches and recorded in the "Performance of algorithms.csv" file in the "results" directory. Even though all the three approaches delivered similar performance, I chose RFE approach, as the test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression. But the disadvantage is, model showed slight underfitting.

Next, tried with various tree based models, performed hyper parameter tuning using the Randomized SearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model, recorded that info into a "relevant_features_by_models.csv" file into the "results" directory. Built models, evaluated on both the training and testing data and recorded the performance metrics in the "Performance of algorithms.csv" file in the "results" directory.
Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor. Saved these two models into the "models" directory.

# Deployment: 
Deployed the XGBoost regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML5.
At each step in both development and deployment parts, logging operation is performed which are stored in the development_logs.log and deployment_logs.log files respectively.

So, now we can find the Concrete compressive strength quickly by just passing the quantities of the raw materials as an input to the web application 😊.

# Web Deployment
Deployed on web using Heroku (PaaS) url:- https://ccs-predictor.herokuapp.com/

# CICD Pipeline
We used Circle-CI and Docker hub for Contineous Integration and Contineous Development.

# Screenshots
![deploy](https://user-images.githubusercontent.com/76841427/150670336-d0ec8926-1a62-4722-af15-7cc5ddc201aa.PNG)

# Tools and technologies used
![tech](https://user-images.githubusercontent.com/76841427/150670346-b5775fd4-f4b3-4731-aa19-c9ec03bc5443.PNG)
