# -*- coding: utf-8 -*-
"""
This is example code for the Random Forest Models built in the manuscript entitled
"Place name here"

@author: Sarah
"""
#Import required functions and/or packages.  This will also occur throughout the example code.
import pandas as pd
import numpy as np

#Load in data to be analyzed.
df = pd.read_excel(r'C:\location_of_data/name_of_excell_datafile.xlsx', sheet_name='name')

#Increase default number of columns and rows to display to better see outputs
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 60)

#Assess data a bit
print(df.head())
df.describe()

#Check to see if there is missing data
print(df.isnull().sum())

#Eliminate any data not to be used, clean data, check for correlations, etc.

#Create testing and training dataset after seperating target variable from features variables
#Import required functions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

#Define the number (k) for k-folds
kf = KFold(shuffle=True, n_splits=10)

#Need to create a seperate dataframe of just the binary target values
X = df.iloc[:, 0:58]  #create dataframe without target, [rows, columns]
y=df['Student Success'] #target variable for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=50) #70% of data training, 30% of data testing

#Check sizes of arrays to make sure it they match each other
print('Training Variables Shape:', X_train.shape)
print('Training Target Shape:', y_train.shape)
print('Testing Variables Shape:', X_test.shape)
print('Testing Target Shape:', y_test.shape)


#We are also going to try to determine the best parameters for the model.
#To do so, we will optimize the parameters (aka hyperparameter tuning) and will include cross validating to help mitigate overfitting the model.
#Will use K-fold cross validation (CV) method for our dataset.

kf = KFold(shuffle=True, n_splits=10)

#Build base model without any changes to default settings
forest_base = RandomForestClassifier()

#Fit data to model via.fit
forest_base.fit(X_train, y_train) #using training data
y_predictions = forest_base.predict(X_test) #Make predictions using testing data set
y_true = y_test #True values of test dataset

#Evaluate the model
#import required functions for model evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

def classification_report_with_accuracy_score_AUC(y_true, y_predictions):
    print(classification_report(y_true, y_predictions))
    print(roc_auc_score(y_true, y_predictions), "ROC_AUC")
    print(accuracy_score(y_true, y_predictions), "accuracy score")
    return accuracy_score(y_true, y_predictions)


#Nested CV
nested_score = cross_val_score(forest_base, X, y, cv=kf, \
                               scoring = make_scorer(classification_report_with_accuracy_score_AUC))
print(nested_score)
print(nested_score.mean())  #97.96%


#Confusion matrix with results, make sure run code all together
confuse_matrix = confusion_matrix(y_true, y_predictions)
sb.heatmap(confuse_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)

#Determine the features of importance that contribute to the model
base_feature_imp = pd.Series(forest_base.feature_importances_, index=X.columns)
base_feature_imp = base_feature_imp.sort_values(ascending=False)
print(base_feature_imp)

#Assess hyperparamters to try to improve upon base model:
#First create our hyperparameter grid
# Number of trees to be included in random forest
n_trees = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 50)]  #This code will create 50 trees b/w 100 and 2000

# Number of features to consider at every split
max_features = ['sqrt','None']  #auto will consider the max features = sqrt(n_features) wheras none will consider all features

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
max_depth.append(None) #Also adding none which means that the nodes are expanded until all leaves are pure or until all leaves contain less than the min_samples_split samples.

# Minimum number of samples required to split a node
min_samples_split = [2, 4, 6, 8, 10]  #default is 2

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6, 8, 10] #default is 1

# Method of selecting samples for training each tree to include using bootstrap method AND to also then try without bootsrap meaning the whole dataset is used to build each tree.
bootstrap = [True, False]

# Create the random grid
hyper_grid = {'n_estimators': n_trees,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(hyper_grid)

#Initiate base model to tune
best_params = RandomForestClassifier(random_state=(25))

#Use random grid search to find best hyperparameters, uses kfold validation as cross validation method
#Search 200 different combinations

best_params_results = RandomizedSearchCV(estimator=best_params, param_distributions= hyper_grid, n_iter=200,
                                         cv=kf, verbose=5, random_state=(30))

#Fit the random search model
best_params_results.fit(X_train, y_train)

#Find the best parameters from the grid search results
best_params_results.best_params_

#Build another hyperparameter grid using narrowed down parameter guidelines from above
#Then use GridSearchCV method to search every combinatino of grid
new_grid = {'n_estimators': [1500, 1600, 1700, 1767, 1800, 1900, 2000],
               'max_features': ['sqrt'],
               'max_depth': [30, 35, 40, 45, 50, 55, 60, 65, 70],
               'min_samples_split': [4, 6, 8, 10, 12],
               'min_samples_leaf': [1, 2, 3],
               'bootstrap': [False]}
print(new_grid)

from sklearn.model_selection import GridSearchCV

best_params2 = RandomForestClassifier(random_state=(30))
best_params2_grid_search = GridSearchCV(estimator=best_params2, param_grid=new_grid, cv=kf, n_jobs=-1, verbose=10)
best_params2_grid_search.fit(X_train, y_train)

best_params2_grid_search.best_params_

###Example Results to use for new model
#{'bootstrap': False,
# 'max_depth': 30,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 1,
# 'min_samples_split': 6,
# 'n_estimators': 1500}

#Using above results specify random forest parameters
final_grid = {'n_estimators': [1500],
               'max_features': ['sqrt'],
               'max_depth': [30],
               'min_samples_split': [6],
               'min_samples_leaf': [1],
               'bootstrap': [False]}

best_params2 = RandomForestClassifier(random_state=(30))
best_params2_grid_search = GridSearchCV(estimator=best_params2, param_grid=final_grid, cv=kf, n_jobs=-1, verbose=10)
best_params2_grid_search.fit(X_train, y_train)

best_params2_grid_search.best_params_


#Best model based upon grid
best_grid_model = best_params2_grid_search.best_estimator_
best_grid_model.fit(X_train, y_train)

y_predictions = best_grid_model.predict(X_test) #Make predictions using testing data set
y_true = y_test #True values of test dataset


#Nested CV
nested_score = cross_val_score(best_grid_model, X, y, cv=kf, \
                               scoring = make_scorer(classification_report_with_accuracy_score_AUC))
print(nested_score)

#Mean values for each parameter

score_accuracy_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring='accuracy').mean()
print(score_accuracy_mean)

score_auc_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring = 'roc_auc').mean()
print(score_auc_mean)

score_precision_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring='precision').mean()
print(score_precision_mean)

score_recall_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring = 'recall').mean()
print(score_recall_mean)

score_f1_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring='f1').mean()
print(score_f1_mean)

scoring = make_scorer(recall_score, pos_label=0)
score_specificity_mean = cross_val_score(best_grid_model, X, y, cv=kf, scoring = scoring).mean()
cross_val_score(best_grid_model, X, y, cv=kf, scoring = scoring)
print(score_specificity_mean)


#Confusion matrix with results, make sure run code all together
confuse_matrix = confusion_matrix(y_true, y_predictions)
sb.heatmap(confuse_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)


#Most important features from best model
feature_imp = pd.Series(best_grid_model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)
feature_imp

#Plot the features of importance
importances= feature_imp
indices = np.argsort(importances)

feature_imp.nlargest(10).plot(kind='bar', title="Top 10 Most Important Features", style='seaborn-colorblind')


#Visualize some trees
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import graphviz
from sklearn import tree

Tree_2 = tree.export_graphviz(forest_base.estimators_[2], out_file =None, filled=True)
graph = graphviz.Source(Tree_2, format='png')
graph




















import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


