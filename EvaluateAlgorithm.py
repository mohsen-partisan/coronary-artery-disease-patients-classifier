import pandas as pd
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
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

    def up_sampling_with_repeating(self, data_train):
        major = data_train[data_train.target == 0]
        minor = data_train[data_train.target == 1]
        minor_upsampled = resample(minor,
                                     replace=True,  # sample with replacement
                                     n_samples=len(major),  # match number in majority class
                                     random_state=27)
        upsampled = pd.concat([minor_upsampled, major])
        upsampled.target.value_counts()
        return upsampled

    def up_sampling_with_SMOTE(self, data_train):
        target_train = data_train.target
        data_train = data_train.drop(['target'], axis=1)
        sm = SMOTE(random_state=27, ratio=1.0)
        data_train, target_train = sm.fit_sample(data_train, target_train)
        x = pd.concat([pd.DataFrame(data_train), pd.Series(target_train).to_frame().T])
        return x

    def cross_validation(self, features):
        for train_index, test_index in kfold.split(features, features.target, None):
            test = features.iloc[test_index]
            target_test = test.target
            test = test.drop(['target'], axis=1)
            train = features.iloc[train_index]
            # self.train_model(train)

            # upsampled_train_with_repeating = self.up_sampling_with_repeating(train)
            # self.train_model(upsampled_train_with_repeating)

            # smote here
            upsampled_train_with_SMOTE = self.up_sampling_with_SMOTE(train)
            self.train_model(upsampled_train_with_SMOTE)

            result = model.score(test, target_test)
            predicted = model.predict(test)
            report = classification_report(target_test, predicted)
            matrix = confusion_matrix(target_test, predicted)
            print(("Accuracy: %.3f%%") % (result * 100.0))
            print(report)
            print(matrix)

    def train_model(self, data_train):
        target_train = data_train.target
        data_train = data_train.drop(['target'], axis=1)
        trained_model = model.fit(data_train, target_train)
        result_train = accuracy_score(target_train, trained_model.predict(data_train))
        matrix_train = confusion_matrix(target_train, trained_model.predict(data_train))
        print(("Accuracy for train: %.3f%%") % (result_train * 100.0))
        print(matrix_train)


    def down_sample(self):
        features_with_target = pd.concat([features, EvaluationAlgorithm().data['target']], axis=1)
        major = features_with_target[features_with_target.target == 0]
        minor = features_with_target[features_with_target.target == 1]
        major_downsampled = resample(major,
                                     replace=True,  # sample with replacement
                                     n_samples=len(minor),  # match number in majority class
                                     random_state=27)
        downsampled = pd.concat([major_downsampled, minor])
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
# models.append(( ' RF ' , randomForest))
# models.append(( ' ADA ' , adaBoost))
# models.append(( ' GB' , gradientBoosting))
models.append(( ' Voting' , voting))
# models.append(( ' xgboost' , XGBClassifier()))
# models.append(( ' mlp' , MLPClassifier()))
results = []
names = []

# complete data
features = DataPreprocessor().select_features()

for name, model in models:
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
    EvaluationAlgorithm().cross_validation(features)





a = 1
