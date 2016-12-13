
#Importing Packages
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from adacost import AdaCost

#Reading Data
train = pd.read_csv("data/claims_train.csv")
test = pd.read_csv("data/claims_test.csv")

#Defining Cost Matrix (rows are predicted classes and columns are true classes)
pmatrix = np.array([[0.0,2.0,4.0,6.0,8.0],
                    [1.0,0.0,2.0,4.0,6.0],
                    [2.0,1.0,0.0,2.0,4.0],
                    [3.0,2.0,1.0,0.0,2.0],
                    [4.0,3.0,2.0,1.0,0.0]])

#Cost Calculation Function
def cost_calc(y_p,y,print_result = False):
    con_mat = confusion_matrix(y_p,y)
    cost_mat = np.multiply(con_mat,pmatrix)
    cost = np.sum(np.multiply(con_mat,pmatrix))/len(y) 
    if print_result:
        print "Confusion Matrix\n",con_mat
        print "Costs\n",cost_mat
        print "Total Cost = ", cost
    else:
        return cost

#Define Evaluation Metric for Grid-Search
score = make_scorer(cost_calc,greater_is_better = True)

#Preparation for modeling
X_train = train.drop('bucket2009',axis = 1)    
y_train = train['bucket2009']
X_test = test.drop('bucket2009', axis = 1)
y_test = test['bucket2009']

#Baseline Model
cost_calc(test['bucket2008'],test['bucket2009'])                    

#Random Forest
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cost_calc(y_pred,y_test)

#Adaboost Classifier
clf = AdaBoostClassifier(algorithm = "SAMME.R",learning_rate=0.1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cost_calc(y_pred,y_test)

#Adacost - SAMME.R
clf = AdaCost(algorithm = "SAMME.R", learning_rate = 0.1, cost_matrix = pmatrix)
clf.fit(X_train,y_train) 
y_pred = clf.predict(X_test)
cost_calc(y_pred,y_test,True)

#Adacost - SAMME.R
clf = AdaCost(algorithm = "SAMME", learning_rate = 0.1, random_state = 100,  n_estimators = 20)
clf.fit(X_test,y_test) 
y_pred = clf.predict(X_train)
cost_calc(y_pred,y_train,True)

#Adacost with Grid-Search
clf = AdaCost(algorithm = "SAMME.R", cost_matrix = pmatrix, random_state = 100)
cv = GridSearchCV(clf,
                  param_grid = {'max_depth':[1,2,3],
                                'n_estimators':[50,100],
                                'learning_rate':[0.01,0.1,1]},
                  scoring = score,
                  verbose = 10, 
                  n_jobs = 1)       

cv.fit(X_train,y_train)
cv.grid_scores_
cv.best_params_,cv.best_score_
clf_cv = cv.best_estimator_
y_pred = clf_cv.predict(X_test)
cost_calc(y_pred,y_test,True)


    
   


                

