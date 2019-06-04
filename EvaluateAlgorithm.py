from Test import getData
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from FeatureSelection import FeatureSelection
import numpy
from Util import Util

class EvalutionAlgorithm:

    def __init__(self):
        self.data = getData()
        self.final_headers = []

    def split_out_train_test(self):
        standardized_array = self.standardize()
        # extract target from array
        target = standardized_array[:, (standardized_array.shape[1] - 1)]
        # extract selected features
        features = self.select_features()
        validation_size = 0.20
        seed = 7
        return train_test_split(features, target, test_size=validation_size, random_state=seed)


    def standardize(self):
        # move age column position next to another continious columns
        age_column = self.data.pop('Age_On_Admission')
        x = self.data.pop('D41stLesionPCITreatedwithStentStentdiameter')
        y = self.data.pop('D41stLesionPCITreatedwithStentStentlenght')
        self.data.insert(2, 'Age_On_Admission', age_column)
        self.data.insert(3, 'D41stLesionPCITreatedwithStentStentdiameter', x)
        self.data.insert(4, 'D41stLesionPCITreatedwithStentStentlenght', y)
        self.final_headers = list(self.data)
        array = self.data.values

        # standardize continious columns
        scaler = StandardScaler().fit(array[:, 0:5])
        array[:, 0:5] = scaler.transform(array[:, 0:5])

        return array

    # feature selection
    def select_features(self):
        feature_selection = FeatureSelection()
        selected_features = feature_selection.selected_features_by_xgboost(evaluate.standardize())
        util = Util()
        selected_features_index = util.selected_features(selected_features)
        selected_data = evaluate.data.ix[:, selected_features_index]
        return selected_data.values


evaluate = EvalutionAlgorithm()
features_train, features_test, target_train, target_test = evaluate.split_out_train_test()



num_folds = 10
seed = 7
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
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, features_train, target_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# evaluate = EvalutionAlgorithm()
array = evaluate.standardize()
features = array[:, 0:47]
target = array[:, 47]
xmodel = XGBClassifier()
xmodel.fit(features, target)
# # plot_importance(xmodel)
# pyplot.bar(range(len(xmodel.feature_importances_)), xmodel.feature_importances_)
# pyplot.xticks(numpy.arange(47), numpy.arange(47))
# pyplot.show()
# print(xmodel.feature_importances_)




a = 1
