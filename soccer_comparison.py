#This program compares the accuracies of several classifiers that predict the outcome of EPL matches with only team names as input.

import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df = pd.read_csv('soccer_cleaned_dummies.csv')
hometeam = df.loc[:,'HomeTeam_Aston Villa' : 'HomeTeam_Wolves']
awayteam = df.loc[:,'AwayTeam_Aston Villa' : 'AwayTeam_Wolves']
AST = df['AST'] #Away shots on target
HST = df['HST'] #Home shots on target
AS = df['AS'] #Away shots
HS = df['HS'] #Home shots
X = pd.concat([hometeam,awayteam],axis=1)
y = df['FTR']
X_train, X_test, y_train, y_test = train_test_split(X,y)



model = MLPClassifier(random_state=0, max_iter = 1000).fit(X_train, y_train)
print("MLP classifier: ", model.score(X_test,y_test))

model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("Gradient Boosting Classifier: ", model.score(X_test,y_test))

model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
print("Random Forest: ", model.score(X_test,y_test))

model = LogisticRegression(random_state=0,max_iter=500,solver="newton-cg").fit(X_train, y_train)
print("Logistic Regression: ", model.score(X_test,y_test))

model = GaussianNB().fit(X_train, y_train)
print("Gaussian NB: ", model.score(X_test,y_test))

model = GaussianProcessClassifier(random_state=0).fit(X_train, y_train)
print("Gaussian Process Classifier: ", model.score(X_test,y_test))

#kernel = 1.0 * RBF(1.0)
#model = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
#print("Gaussian Process Classifier with RBF: ", model.score(X,y))
