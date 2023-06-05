# SyriaTel Customer Churn Prediction
Sources
Canvas - Flatiron School https://learning.flatironschool.com/courses/6400

Flatiron Github https://github.com/flatiron-school/NTL01-DTSC-LIVE-040323/tree/main/3phase

OVERVIEW 
To build a Machine Learning model for SyriaTel, a telecommunications company that predicts whether a customer will discontinue their subscriptions to the service provided. SyriaTel has compiled data about their customers who has retained and not retained their service, thus churning. Having a better understanding of the data, we can build visuals and see if there is any insights we can get before deciding how to build our Machine Learning model, and tune if necessary.

Business Problem 
Churning in the telecommunications industry is problematic for the provider as their customers offset their OPEX. The more churn the provider encounters, the company can be at a deficit considering their overhead expenditures to maintain operations.

According to SyriaTel's data, its been noted that the provider is suffering from a 14.49% churn over rate. The high churn rate can be attributed for the following reasons listed below:

Tenure
Service plan a customer has
Calls made throughout the day and to customer service
Charges incurred throughout the day
Without any understanding or guidance into these issues, the provider can be at a loss, thus having to slow down business.

Business Objectives 
The objective in conducting the analysis is to create a sophisticated Machine Learning algorithm that can predict customer churn, develop insighful strategies to prevent future customer turnover and possibly increase retention for SyriaTel.

Understanding the Data
We can see how the information is displayed in the DataFrame.

Visualization

Rate of Churn by Area Code

This is a bar plot showing the Rate of Churn by Area Code. We can see area code '415' is suffering from high churn in comparison with the rest of the area codes. This can be in result of having a lot of customers in '415', assuming the rate of churn is consistent across the other area codes.
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/abb33b03-48c7-41e5-8c8e-f241b7858564)

Rate of Churn - Retained customers vs Churned customers

This bar plot shows the rate of churn in the DataFrame. We can see the Churn rate is at about 15% whereas the Retained customers are at around 85%.
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/790033af-aa45-4f2f-864f-55390f41697a)

Count of Churn by State

The bar plot shows the Count of Churn by State. The plot shows all of the states where the customers are located.

We can see the count of retained customers and churned. It shows churned counts descending in order. Its clear from the left side that the states are having the most churns, starting with Texas as it leads.
The state with the most retained customers is West Virginia, topping around 95 customers.
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/7f42b61a-91ea-4062-97ba-aa93b40cdff1)

Top 5 (Churn by State)
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/470b2194-474c-4397-8566-b15e59fd9253)

Churn by Number of Customer Service Calls
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/c07ddc7f-fd85-4bf5-95e8-020aab71cbb1)

Data Modeling Preparation
0. Categorical/Numerical
        ______________________Distinguish Categorical / Numerical______________________ 
To help better organize the data and feed it directly into the model, we need to understand the Categorical and Numerical columns we want to further explore.

The categorical columns are those with various categories given a numerical value that can be interpreted by the algorithm, when One Hot Code is applied.

The numerical columns can be standardized to avoid any bias in the predictions due to different ranges in the training data

1. One-Hot Encoding
 __________________One Hot Encode__________________
One-hot encoding is a technique used in machine learning to convert categorical data into a format that can be fed into machine learning algorithms to improve prediction accuracy.
In this case, we are feeding categorical variable that consist of columns 'state, 'international_plan', 'voice_mail_plan', and 'area_code'.

2.  ______________________StandardScaler()______________________
Applying StandardScaler() to the DataFrame, we can resize the distribution of values, to combat any differences in ranges and units of measure. This is an important step of data preparation before processing it to the algorithm.

In this case, we are passing the numerical variable through StandardScalers fit_transform() method so it can resize the numerical columns, which in this case are: 'account_length', 'number_vmail_messages', 'total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_intl_calls', 'customer_service_calls'

3.   __________________Define X and y variables __________________
To fit the data into a learning model, we need to convert the categories into dummy variables, which is denoted as binary in this case (0, 1). So to use the predictors (X-variable) in a model, for this case, a logistic regression, we must have dummy variables.

4.          _____________________Train-Test Split_____________________
Making use of the Train-Test Split is vital as its used to estimate the performance of the learning model for predictions, when comparing the algorithms for the predictive model at hand.

5.  ____________________________RESAMPLING - USING SMOTE____________________________
                                                    SMOTE
We use the SMOTE, which stands for Synthetic Minority Oversampling Technique. This is a algorithm design to oversample the minority class. Since there is an imbalance in class 1 (True), we will need to apply SMOTE to generate synthetic data points that is based off the original data points.

After applying SMOTE, the Synthetic sample class distribution is now 50/50, meaning the distribution of data points are now balanced.

6.  ________________________Logistic Regression________________________
This classification model is used to predict a binary outcome based on prior data observations, well suited for predicting customer churn (0, 1). Using churn as the dependent variables, y, we are able to see the relationships with the independent variables in X.


Model Evaluation
Logistic Regression (Stock) 
To get a better indication of the performance of the Logistic Regression Model, we will define a function called model_evaluation, that will calculate:

  - Accuracy
  - Recall
  - Precision
  - F1-Score
Then, it will plot a Confusion Matrix for the trained and test data, and result with a summary at the bottom of each matrix, showing us the values of the respective metrics listed above.

● ● ● Logistic Regression Model Tune-up 
At this stage, the Logisitc Regression model will be tuned via hyperparameters, which in result can reach higher performance.

To tune the hyperparameters:

The Pipeline needs to be defined with the max iterations and tolerance.
Define the hyperparameter grid used for tuning
Fit the grid search cross validation
Plug in the best parameters and score
Retrieve teh best model

_________________Hyperparameter Tuning_________________
This code is used to perform hyperparameter tuning on a logistic regression model using GridSearchCV. It first line creates a pipeline that contains a logistic regression model with the specified maximum number of iterations and tolerance. Then, itcreates a dictionary of hyperparameters to tune. The ‘logreg__’ prefix is used to specify that the hyperparameters are for the logistic regression model in the pipeline.

Then, a GridSearchCV object is created with the pipeline, hyperparameters, 5-fold cross-validation, and accuracy as the scoring metric. Its then fitted with the GridSearchCV object to the training data.

The code displays the best hyperparameters and best score from the GridSearchCV object.

Finally, the best estimator is extracted from the GridSearchCV object.

_______________Feature Importance_______________
This code is used to plot the top 10 features of the best logistic regression model. The codeextracts the coefficients of the best logistic regression model. Then, it normalizes the feature importance values to a scale of 0-100 and select the top 10 features.

The names of the features are retrieved and a function is called that plots the top 10 features.
![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/7dc2061b-1b4f-4e4c-98f8-20410e09bc1d)


____________Evaluation ____________
Model Evaluation

Results:

Stock

Training Data Results:
Accuracy: 0.9091546006539001
Precision: 0.9332344213649851
Recall: 0.8813638486688463
F1-score: 0.9065577708383378

Test Data Results:
Accuracy: 0.8525179856115108
Precision: 0.5113636363636364
Recall: 0.36
F1-score: 0.4225352112676056



● ● ● Tune-up

Training Data Results:
Accuracy: 0.9105558150397011
Precision: 0.9452887537993921
Recall: 0.8715553479682392
F1-score: 0.9069258809234508

Test Data Results:
Accuracy: 0.8441247002398081
Precision: 0.4657534246575342
Recall: 0.272
F1-score: 0.3434343434343434

The Logistic Model ● ● ● Tune-up has improved over the Stock , by increasing in Accuracy and Precision, and a decrease in Recall. What this suggest is that there are fewer false positives relative to true positives, and when the model predicts a positive outcome, it is more likely to be correct.

As for the decrease in Recall in the ● ● ● Tune-up model, we can say there might be a slight overfitting in the model that causes the test to perform poorly.



Decision Trees 𓆱𓍊𓋼𓍊𓋼𓍊
For the second model used, we will go with Decision Trees. It is a good algorithm that can demonstrate various outcomes. It is great for decision making at it shows all possible outcomes to a problem. We can use this to predict various outcomes in the future for churn.

The below we can fit a Decision Tree classifier on the resampled training set, then predict on the test set.
We implement Feature Importance on a Decision Tree Classifier and plotted the top 10 features with their importance value.

![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/5c7e90eb-7bd5-4bf6-8f04-68fabd01ef18)

■ ■ ■ Decision Model Tune-up 
At this stage, the Decision Tree model will be tuned via hyperparameters, which in result can reach higher performance.

To tune the hyperparameters:

The Pipeline needs to be defined with the max iterations and tolerance.
Define the hyperparameter grid used for tuning
Fit the grid search cross validation
Plug in the best parameters and score
Retrieve the best model

Decision Tree Model Evaluation

Results:

Stock

Training Data Results:
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1-score: 1.0

Test Data Results:
Accuracy: 0.854916067146283
Precision: 0.5123456790123457
Recall: 0.664
F1-score: 0.578397212543554



■ ■ ■ Tune-up

Training Data Results:
Accuracy: 0.9848201774871556
Precision: 0.9905482041587902
Recall: 0.9789817842129845
F1-score: 0.9847310312426591

Test Data Results:
Accuracy: 0.815347721822542
Precision: 0.4152046783625731
Recall: 0.568
F1-score: 0.4797297297297297

The Logistic Model ■ ■ ■ Tune-up has improved over the Stock , by increasing in Accuracy and Precision, and a decrease in Recall. What this suggest is that there are fewer false positives relative to true positives, and when the model predicts a positive outcome, it is more likely to be correct.

As for the decrease in Recall in the ■ ■ ■ Tune-up model, we can say there might be a slight overfitting in the model that causes the test to perform poorly.

When comparing the performance between the training and the test data, we can agree the training data is performing exceptionally well, but its too good to be true, hence overfitting is a possibility. As for the test data, it indicates an accuracy of 85.49%. Recall/Precision/F1-Score is not bad nor good, but there is still room for more improvement.



![image](https://github.com/anvadev/Phase_3_Project/assets/50537930/122b1bfd-c496-48c1-b936-5153a8adc380)









