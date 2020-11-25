
#Sections 4-4.3 of Intro to Statistical Learning...

#Logistic Regression a method for Classification
    #Classification is the problem of identifying to which of a set of categories a new observation belongs to, baed on training data
        #Some examples:
        #Spam vs 'Ham' emails
        #Loan Default - (yes or no)
        #Disease Diagnosis
        #These are Examples of Binary Classification - two classes

#Logistic Regression allows us to solve classification problems, where we are trying to predict discrete categories
#The convention for binary classification is to have two categories - 0 and 1
#Sigmoid - Logistic Function - key to understanding logistic regression to perform a classification
    #Output values from 0 to 1
    #Based on this probability, we assign a cut-off point 

#After traning the LogReg model on training data, we will evaluate it's performance on some test data
#We can use a confusion matrix to evaluate classification models
#Type 1 error - false positives - actually negative, looks positive
#Type 2 error - false negatives - actually positive, looks negative

#-------------------------------------------------------------------

#Titanic Data Set - Exploratory Data Analysis/Visualization

#Trying to predict the classification-- survival or deceased, for passengers who were on the Titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
#print(train.head())

#print(train.isnull())

#Quick heatmap check to see what data we are missing, i.e. Null values
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#plt.show()

#Exploring different data-visualization 
sns.set_style('whitegrid')

#First metric, getting a count of who survived and who didn't survive
# sns.countplot(x='Survived', data=train)
# plt.show()

#Getting count based on gender
# sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')
# plt.show()

#Getting count based on Passenger Class
# sns.countplot(x='Survived', data=train, hue='Pclass', palette='RdBu_r')
# plt.show()
 
#Seeing the age of passengers on the titanic 
#sns.distplot(train['Age'].dropna(), kde=False, bins=30)
# train['Age'].plot.hist(bins=35) #another method
#plt.show()

#Seeing the no. of siblings/spouse people had on board 
# sns.countplot(x='SibSp', data=train)
# plt.show()

#Seeing the price people paid
# train['Fare'].hist(bins=40, figsize=(10,4))
# plt.show()

#Doing it in interactive way with cufflinks and plotly
# import cufflinks as cf
# cf.go_offline()
# train['Fare'].iplot(kind='hist', bins=30)
# plt.show()


#----------------------------------------------------------------------

#Cleaning our data - missing data

#Age column has some missing fields - one way to fill them is using the mean - this is known as imputation
#We can be a step smarter, by checking average age by Passenger class

# sns.boxplot(x='Pclass', y='Age', data=train)
# plt.show()

#Creating function to fill in missing values in Age column  
def impute_age(cols):

    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
#apply is to create custom function and apply it
#axis is 0 by default, and it refers to rows
#we want to apply to the columns so axis = 1

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

#Since there are so much missing info in the cabin, we are going to drop it
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

#---------------------------------------

# We did the first step of cleaning data
# Now we need to deal with categorical features, we'll need to convert categorical features into 'numerical dummy values' using pandas,
#... otherwise out ML algorithm won't be able to directly take in categorical features as inputs. Needs numerical inputs

#Converting categorical variables into dummy or indicator variables
pd.get_dummies(train['Sex'])
#pass in the col we want to convert.
#If female is true, then male is definitely false, so one col can perfectly predict the other column, and this poses a problem known as multi-collinearity that can mess up the algorithm
#so we do drop_first=True to drop the first column, here the 'Female' column
sex = pd.get_dummies(train['Sex'], drop_first=True)

#doing same thing here, trying to avoid one column being perfect predictors of another column
embark = pd.get_dummies(train['Embarked'], drop_first=True)

#Now adding these above columns to our dataframe, using concatenation
train = pd.concat([train, sex, embark], axis=1)
print(train.head())

#Dropping columns that we are not going to use
train.drop(['Sex', 'Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True)
print(train.head())


#====================================================================================

#Prediction for classification - whether passengers on Titanic survived or not.
#y is what we are trying to predict, so in this case, it should be the 'Survived' column
#x is everything else but y, so all the columns except the 'Survived' column
x = train.drop('Survived', axis=1)
y = train['Survived']

#Doing train-test split for training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#Now we have to create a model to train and predict
#For logistic regression we do:
from sklearn.linear_model import LogisticRegression

#create an instance of the logisticregression model
logmoel