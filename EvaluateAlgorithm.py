from Test import getData
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class EvalutionAlgorithm:

    def __init__(self):
        self.data = getData()

    def split_out_train_test(self):
        array = self.data.values
        features = array[:, 0:93]
        target = array[:, 93]
        validation_size = 0.20
        seed = 7
        features_train, features_test, target_train, target_test = train_test_split(features, target, validation_size, seed)


split = EvalutionAlgorithm()
split.split_out_train_test()

df[[c for c in df if c not in ['b', 'x']]
       + ['b', 'x']]