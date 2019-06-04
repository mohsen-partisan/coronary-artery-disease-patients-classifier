
from xgboost import XGBClassifier

class FeatureSelection:

    def selected_features_by_xgboost(self, data):
        features = data[:, 0:(data.shape[1]-1)]
        target = data[:, (data.shape[1]-1)]
        xmodel = XGBClassifier()
        xmodel.fit(features, target)
        return list(xmodel.feature_importances_)