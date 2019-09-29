

from  DataPreprocessor import DataPreprocessor
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

class MeanClassFinder:

    def find_instances_near_each_class_centroid(self):
        data = DataPreprocessor().select_all_features()
        # data = data.drop(['target'], axis=1)

        mean_class_0 = data.loc[data['target'] == 0].mean()
        mean_class_1 = data.loc[data['target'] == 1].mean()
        means = pd.concat([mean_class_0, mean_class_1], axis=1)
        means = means.T
        #######
        data = data.append(means, ignore_index=True)
        #######
        class_0 = data.loc[data['target'] == 0]
        class_1 = data.loc[data['target'] == 1]

        #########

        dist_condensed_class_0 = pdist(class_0.values, metric='seuclidean')
        distance_mat_0 = pd.DataFrame(squareform(dist_condensed_class_0), index=class_0.index,
                                    columns=class_0.index)
        distance_class_0 = distance_mat_0.iloc[759]
        distance_class_0.sort_values(ascending=True, inplace=True)
        distance_class_0.drop(labels=[1423], inplace=True)
        sort_by_distance_class_0 = distance_class_0.index
        nearest_to_centroid_class_0 = sort_by_distance_class_0[0:len(sort_by_distance_class_0) // 4]

        dist_condensed_class_1 = pdist(class_1.values)
        distance_mat_1 = pd.DataFrame(squareform(dist_condensed_class_1), index=class_1.index,
                                      columns=class_1.index)
        distance_class_1 = distance_mat_1.iloc[664]
        distance_class_1.sort_values(ascending=True, inplace=True)
        distance_class_1.drop(labels=[1424], inplace=True)
        sort_by_distance_class_1 = distance_class_1.index
        nearest_to_centroid_class_1 = sort_by_distance_class_1[0:len(sort_by_distance_class_1) // 4]

        new_data_class_0 = data.ix[nearest_to_centroid_class_0]
        new_data_class_1 = data.ix[nearest_to_centroid_class_1]
        new_data_centroids = pd.concat([new_data_class_0, new_data_class_1])

        nearest_to_centroids = np.concatenate([nearest_to_centroid_class_0, nearest_to_centroid_class_1])
        not_near_to_centroids = set(data.index) - set(nearest_to_centroids)
        new_data_not_centroids = data.ix[not_near_to_centroids]

        return new_data_centroids, new_data_not_centroids

        s=0
        #########


        # distance_mat_class_0 = pd.DataFrame(distance_matrix(means.values, class_0.values), index=means.index,
        #                             columns=class_0.index)
        #
        # distance_mat_class_0.sort_values(by=0, axis=1, inplace=True)
        # sort_by_distance_class_0 = list(distance_mat_class_0)
        # nearest_to_centroid_class_0 = sort_by_distance_class_0[0:len(sort_by_distance_class_0)//4]
        #
        # distance_mat_class_1 = pd.DataFrame(distance_matrix(means.values, class_1.values), index=means.index,
        #                                     columns=class_1.index)
        # distance_mat_class_1.sort_values(by=1, axis=1, inplace=True)
        # sort_by_distance_class_1= list(distance_mat_class_1)
        # nearest_to_centroid_class_1 = sort_by_distance_class_1[0:len(sort_by_distance_class_1)//4]
        #
        # new_data_class_0 = data.ix[nearest_to_centroid_class_0]
        # new_data_class_1 = data.ix[nearest_to_centroid_class_1]
        # new_data = pd.concat([new_data_class_0, new_data_class_1])
        #
        # return new_data



