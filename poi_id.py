#!/usr/bin/python

import os
from time import time
import re
import numpy as np
import pandas as pd
import sys
import pickle
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#sklearn imports******
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA

from sklearn.grid_search import GridSearchCV

from sklearn import tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
sys.path.append("../tools/")


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### First, I use all the original features, to see which one provides the best
### Precision and recall

#features_list = ['poi', 'to_messages', 'deferral_payments', 'expenses', 
#                 'deferred_income', 'long_term_incentive', 
#                 'restricted_stock_deferred', 'shared_receipt_with_poi', 
#                 'loan_advances', 'from_messages', 'other', 'director_fees', 
#                 'bonus', 'total_stock_value', 'from_poi_to_this_person', 
#                 'from_this_person_to_poi', 'restricted_stock', 'salary', 
#                 'total_payments', 'exercised_stock_options']

### After implementing SelectBest I choose the following features
features_list = ['poi','salary', 'bonus', 'deferred_income', 
              'exercised_stock_options', 'total_stock_value']


### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

    
### How many datapoints in the dataset? There are 146 people in our dataset   
print len(data_dict)

### how many features in the dataset? There are 21 different features
no_of_features = len(data_dict[data_dict.keys()[0]])
print no_of_features

### I print a list of features to choose which ones I will use    
print data_dict["YEAP SOON"].keys()

### How many POIs vs. non-POIs in the dataset?
print len([x for x in data_dict.itervalues() if x['poi'] is True])
## There are 18 "POIs"
print len([x for x in data_dict.itervalues() if x['poi'] is False])
### There are 18 non "POIs"

### Feature Visualization. I am going to plot the "salary" and "bonus" 
### before deleting the outliers

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.ylim(0, 100000000)
plt.xlim(0, 30000000)
plt.xlabel("salary (in 10000000)")
plt.ylabel("bonus (in 100000000)")
plt.title ("Relatioship between salary and bonus")
plt.show()

#------------------------------------------------------------------------------

### Task 2: Remove outliers

for keys, values in data_dict.items():
    print keys
### There are two values that are not names (outliers), so we proceed to delete 
### them
del data_dict ["THE TRAVEL AGENCY IN THE PARK"]
del data_dict ["TOTAL"]

### If we print them again, they are no longer there
for keys, values in data_dict.items():
    print keys

### Let's print the data lenght without the outliers
print len(data_dict)

### How many datapoints do we have no? Now we have 144 names in the dataset
print len(features_list)

### I am going to plot the "salary" and "bonus" after we deleted the outliers

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.title ("Relatioship between salary and bonus")
plt.show()

### Here we find out how many missing values per feature in our features_list
for x in features_list[1:]:
    print x, len(dict((key, value) for key, value in data_dict.items() 
    if value[x] == 'NaN'))

### In the following code we will loop through the features list to find all 
### the non NaN values, and append them to a list
for x in features_list[1:]:
    features_values = []
    for K in data_dict.values():
        if K[x] != 'NaN':
            features_values.append(K[x])
    ## Then, we calculate the mean
    dfmean = np.mean(features_values)
    dfmedian = np.median(features_values)
    ## Print the feature, the mean and the median to see which one we will use
    print x,"mean=",  dfmean, "median=", dfmedian

### After calculating both the mean and the median, I decide to use the median
    for key in data_dict.keys():
        if data_dict[key][x] == 'NaN':
            data_dict[key][x] = dfmedian

### Now we dont see any NaN
for x in features_list[1:]:
    print x, len(dict((key, value) for key, value in data_dict.items() 
    if value[x] == 'NaN'))

#------------------------------------------------------------------------------

### Task 3: Create new feature(s)
   
### I am going to create a new feature (Salary/Bonus ratio)
 
new_features = 'bonus_salary_ratio'
features_list.append(new_features)
### Now, instead of 6 we have 7 features, but it is still empty
print len(features_list)
### Here is the list of features I chose
print features_list[:]

### Then, we calculate the ratios, and include them inside the new feature
for key in data_dict.keys():
    data_dict[key][new_features] = np.true_divide(data_dict[key]["bonus"],
             data_dict[key]["salary"])

### let's print one of the records to ensure that the new ratios are included
### in the features_list
print data_dict ['LAY KENNETH L'].keys()
print data_dict ['GRAY RODNEY']


for x, y in data_dict.items():
    print x, y["bonus"], y["salary"], y["bonus_salary_ratio"]   

### I am going to plot the "salary" and "bonus_salary_ratio"
features = ["salary", "bonus_salary_ratio"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
### plot the new feature
for point in data:
    salary = point[0]
    bonus_salary_ratio = point[1]
    plt.scatter( salary, bonus_salary_ratio )

plt.title ("Relatioship between salary and bonus_salary_ratio")
plt.ylim(0,50)
plt.xlabel("salary")
plt.ylabel("bonus_salary_ratio")
plt.show()

### Store to my_dataset for easy export below. 
my_dataset = data_dict


### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Let's select the best features and remove some. The best scoring value
### is k = 5.

from sklearn.feature_selection import SelectKBest
k = 5
skb = SelectKBest(k=k)
skb = skb.fit(features,labels)
features = skb.transform(features)

top_scores = np.sort(skb.scores_)[-k:]

new_features_list = ['poi']
for i in xrange(len(features_list[1:])):
   if skb.scores_[i] in top_scores:
       new_features_list.append(features_list[1:][i])

features_list = new_features_list

### Let's print the new_features_list.
print new_features_list[:]

#------------------------------------------------------------------------------

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#clf = tree.DecisionTreeClassifier()
#clf = SVC()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#pipeline = make_pipeline(StandardScaler(), RandomizedPCA(), clf)
pipeline = make_pipeline(StandardScaler(), clf)


### cross validation  
cv = StratifiedShuffleSplit(labels, test_size=0.2, random_state=42)

### Adjust the parameters
params = dict()
# for PCA
#params['randomizedpca__iterated_power'] = [1, 2, 3]
#params['randomizedpca__n_components'] = [2, 4, 6, 8, 10]
#params['randomizedpca__random_state'] = [42]
#params['randomizedpca__whiten'] = [True, False]

if str(clf)[0] == 'D':
    params['decisiontreeclassifier__max_features'] = ['auto', 'sqrt', 'log2', None]
    params['decisiontreeclassifier__criterion'] = ['gini', 'entropy']
    params['decisiontreeclassifier__random_state'] = [42]
    params['decisiontreeclassifier__class_weight'] = ['auto', None]
    
if str(clf)[0] == 'S':
    params['svc__random_state'] = [42]
    params['svc__C'] = [2**x for x in np.arange(-15, 15+1, 3)]
    params['svc__gamma'] = [2**x for x in np.arange(-15, 15+1, 3)]
    

grid_search = GridSearchCV(pipeline, param_grid=params, n_jobs=1, cv=cv)

grid_search.fit(features, labels)

clf = grid_search.best_estimator_

#------------------------------------------------------------------------------

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
### StratifiedShuffleSplit.html
test_classifier(clf, my_dataset, features_list )

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
     
#------------------------------------------------------------------------------

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

