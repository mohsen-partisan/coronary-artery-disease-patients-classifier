from DataHandler import getData
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from FeatureSelection import FeatureSelection
import numpy
from Util import Util
from DataPreprocessor import DataPreprocessor

class EvalutionAlgorithm:

    def __init__(self):
        self.data = getData()
        self.final_headers = []

num_folds = 10
seed = 12
scoring = 'accuracy'
models = []
num_trees = 200
# two bagging alg
baggingClassifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=num_trees, random_state=seed)
randomForest = RandomForestClassifier(n_estimators=num_trees)
# two boosting alg
adaBoost = AdaBoostClassifier(n_estimators=30, random_state=seed)
gradientBoosting = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
# Voting ensemble
# create the sub models
estimators = []
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))
estimators.append(('logistic', LogisticRegression()))
voting = VotingClassifier(estimators)
models.append(( ' LR ' , LogisticRegression()))
models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
models.append(( ' KNN ' , KNeighborsClassifier()))
models.append(( ' CART ' , DecisionTreeClassifier()))
models.append(( ' NB ' , GaussianNB()))
models.append(( ' SVM ' , SVC()))
models.append(( ' BC ' , baggingClassifier))
models.append(( ' RF ' , randomForest))
models.append(( ' ADA ' , adaBoost))
models.append(( ' GB' , gradientBoosting))
models.append(( ' Voting' , voting))
models.append(( ' xgboost' , XGBClassifier()))
models.append(( ' mlp' , MLPClassifier()))
results = []
names = []
features = DataPreprocessor().select_features()
standardized_array = DataPreprocessor().standardize()
target=standardized_array[:, (standardized_array.shape[1] - 1)]
for name, model in models:
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, features, target, cv=kfold, scoring=make_scorer(Util.classification_report_with_accuracy_score))
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


a = 1
