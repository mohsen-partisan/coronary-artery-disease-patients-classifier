import pandas as pd
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

from DataHandler import getData
from DataPreprocessor import DataPreprocessor
from Util import Util


class EvaluationAlgorithm:

    def __init__(self):
        self.data = getData()
        self.final_headers = []

    def up_sample(self):
        features_with_target = pd.concat([features, EvaluationAlgorithm().data['target']], axis=1)
        features_with_target.target.value_counts()
        major = features_with_target[features_with_target.target == 0]
        minor = features_with_target[features_with_target.target == 1]
        major_downsampled = resample(minor,
                                     replace=True,  # sample with replacement
                                     n_samples=len(major),  # match number in majority class
                                     random_state=27)
        downsampled = pd.concat([major_downsampled, major])
        downsampled.target.value_counts()
        return downsampled

num_folds = 10
seed = 12
scoring = 'accuracy'
models = []
num_trees = 100
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
# models.append(( ' LR ' , LogisticRegression()))
# models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
# models.append(( ' KNN ' , KNeighborsClassifier()))
# models.append(( ' CART ' , DecisionTreeClassifier()))
# models.append(( ' NB ' , GaussianNB()))
# models.append(( ' SVM ' , SVC()))
# models.append(( ' BC ' , baggingClassifier))
models.append(( ' RF ' , randomForest))
# models.append(( ' ADA ' , adaBoost))
# models.append(( ' GB' , gradientBoosting))
# models.append(( ' Voting' , voting))
# models.append(( ' xgboost' , XGBClassifier()))
# models.append(( ' mlp' , MLPClassifier()))
results = []
names = []
features = DataPreprocessor().select_features()




# standardized_array = DataPreprocessor().standardize()
downsampled = EvaluationAlgorithm().up_sample()
target=downsampled.target
downsampled = downsampled.drop(['target'], axis=1)

for name, model in models:
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, downsampled, target, cv=kfold, scoring=make_scorer(Util.classification_report_with_accuracy_score))
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle( ' Ensemble Algorithm Comparison ' )
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()





a = 1
