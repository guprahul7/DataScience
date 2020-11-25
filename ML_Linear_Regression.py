
#MACHINE LEARNING

#Machine Learning is a method of data analysis that automates analytical model building
#Using algorithms that iteratively learn from data, machine learning allows computers to find hidden insights without being explicitly programmed where to look

#Applications of ML
#   - Fraud Detection, Recommendation Engines, Text Sentiment Analysis
#   - Web search Results, Read-time ads on webpages, Credit scoring, Prediction of equipment failures,
#   - Pattern and Image Recognition, New pricing models, Financial Modeling, Email spam filtering

#Process
#                                 _________-> Test Data -> ______________
#Data_Acqusition -> Data_Cleaning/                                       \
#                                \---- -> Model_Training_&_Building --> Model Testing -> Model Deployment
#                                                           ^<-----------/    

#3 Main Types of Machine Learning 
#   - Supervised Learning -> Have Labeled data and are trying to predict label based on known features
                         #-> Labeled data, input where the desired output is known
                         #-> Through Methods like classification, regression, prediction, gradient boosting, uses patterns to predict values of label on other unlabeled data
                         #-> Applications where historical data predicts likely future events
                         #-> Example: anticipating which credit card transactions are likely to be fraudulent, or which insurance customer is likely to file a claim
                         #-> Example: Can attempt to predict price of a house based on features of house based on historical price data
# 
# - Unsupervised Learning -> Have unlabeled data and trying to group together similar data points based on features
                         #-> Used against data that has n historical labels
                         #-> System is not told the 'right answer'. Algorithm must figure out what is being shown
                         #-> Goal is to explore data and find some structure within
                         #->Popular techniques - self-organizing maps, nearest-neaigbor mapping, k-means clustering, sinular value decomposition
                         #-> Used to segment text topics, recommend items and identify data outliers

#   - Reinforcement Learning -> Algorithm learns to perform an action from experience
                            #-> Often used for robotics, gaming, navigation
                            #-> Algorithm discovers through trial and error which actions yield greatest rewards
                            #-> Has 3 main components: Agent(the learner/decision-maker), Environment(everything the agent interacts with), Actions(what the agent can do)
                            #-> Objective is: for the agent to choose actions that maximize the expected reward over a given amount of time
                            #-> The agent will reach the goal by following the best policy


#ScikitL-Learn strives to have a uniform interface across all methods 
#Given a sklearn estimator object named model, the following methods are available
# On all estimators, you are going to be able to fit training data

# For supervised learning applications, this accepts two arguments: the data X and the labels y (model.fit(X,y))

#For unsupervised learning applications, this accepts only a single argument, the data X (model.fit(X)), (makes sense because unsupervised learning only works with unlabeled data)

#Supervised estimators, you will have predict method (model.predict()), which given a trained model, will predict the label of a new set of data. 
#... this method accepts one argument, the new data X_new (model.predict(X_new),) and returns the learned label for each object in the array



#==================================================================================#

# -- LINEAR REGRESSION THEORY -- 

#Francis Galton
# "A father's son's height tends to regress (or drift towards) the mean(average) height" 

# -- Regression Line - draw a line that's close to as many points as possible
# -- CLassic Linear Regression - (or Least Squares Method), you only measure the closeness in the "up and down" direction
#This method is fitted my minimizing the "sum of squares of the residuals"
# -- The residuals, for an observation, is the difference between the observation (the y-value) and the fitted line

 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('USA_Housing.csv')

#To get a summary overview of the dataframe - useful
print(df.info()) 

#To get a statistical summary of the dataframe - useful
print(df.describe()) 

#To get a list of all column names - useful
print(df.columns) 
#print(df.loc[:5]['Avg. Area Number of Bedrooms'])

#Quick Check to see distribution of a column
sns.distplot(df['Price'])
plt.show()
plt.tight_layout()

#Quick Check for Heatmap - Need to get correlation matrix, which is the a matrix with values of one column in terms of another (like Row-Echelon), for all the columns
print(df.corr())
sns.heatmap(df.corr(), annot=True)
plt.show()


#==================================================================================#

# -- LINEAR REGRESSION -- 

#Check out Chapters 2 & 3 in Intro-to-Statistical-Learning by Gareth James - reference book, pdf available free online 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('USA_Housing.csv')


#----- STEP 1 ---- training variables and target variable
#The first thing we need to do is split the data into an 
#X array - for training, and
#Y-array - the target variable - which is what we are trying to predict, in this case the price column,

#disregarding 'address' column for our model as it contains only text which is not needed here
#print(df.columns)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

Y = df['Price']

#----- STEP 2 ---- Train/Test split -
#we want to have a training set for the model, and a testing set in order to test the model once it's been trained
#SKlearn comes with train/test split function

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
#Tuple un-packing happening here
#test_size = the % of the dataset that we want to allocate to the test_size (0.4 = 40%)
#random_state is basically like seed

#-- NOW WE TRAIN OUR MODEL --
from sklearn.linear_model import LinearRegression

#Since we have imported the model, we go ahead and instantiate an instance of the model
lm = LinearRegression()

#Use method .fit() to train or fit  my-model/my-training-data
lm.fit(X_train, Y_train)
print(lm.fit(X_train, Y_train))

#-- EVALUATE -- 
# Next step is to evaluate our model by checking out it's coefficients and seeing how we can interpret them
print('\n', ' \"This is the intercept\"')  
print(lm.intercept_) #Checking the intercept

#Checking coefficients 
print('\n', ' \"Array of coefficients\"') 
print(lm.coef_,) #returns an array of coeffcients
#Each coefficient in the array relates to the columns in X or X-train (X_train.columns)

#Now creating a dataframe off of coefficients and the columns to see relation b/w them
cdf = pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coeff'])
print('\n \"This is the dataframe with columns and their respective coefficients\"\n')
print(cdf.head())

# #seeing real-life sample-data stored in sklearn
# from sklearn.datasets import load_boston
# boston = load_boston() 
# print(boston.keys()) #because boston is a dictionary with values


#==================================================================================#


#----- PART 2 -----
#Above we fit the model to our training set, now we want to grab predictions from our test set
#We want to pass in a set that the model hasn't seen before, so we use X-test 
predictions = lm.predict(X_test) 
print('\n', '\"Array of predictions\"') 
print(predictions)
#returns array with the predicted Y values (here, the prices of the house)

#Since we did the train-test split, we know that Y_test contains the correct prices of the house
#So we wanna check how far off are the predictions from the test/actual prices
#One quick way to analyze this is by using scatter plot
# plt.scatter(Y_test, predictions)
# plt.show()  #A perfectly straight line is ideal, 

#Creating histogram of the distribution of our residuals (Residuals are the difference b/w actual-values and predicted-values)
sns.distplot((Y_test - predictions))
#If the residual is normally distributed - then it means your LinearRegression model was a correct choice for the data
#If it's not normally distributed or wierdly distributed, we should try another model for the data
#plt.show() 

#--- EVALUATION METRICS -- 
#There are 3 common evaluation metrics for regression problems
#   -> Mean Absolute Error (MAE) - The mean of the absolute values of the errors
#   -> Mean Squared Error (MSE) - The mean of the squared errors
#   -> Root Mean Squared Error (RMSE) - The square-root of the mean of the squared-errors, i.e. 
#       RMSE = sqrt(MSE)
# All of these are loss functions because we want to minimize them

#Calculating Evaluation Metrics
from sklearn import metrics

mae = metrics.mean_absolute_error(Y_test,predictions) 
print('\n \"MAE\": ', mae )

mse = metrics.mean_squared_error(Y_test, predictions)
print('\n \"MSE\": ', mse )

rmse = np.sqrt(mse)
print('\n \"RMSE\": ', rmse )