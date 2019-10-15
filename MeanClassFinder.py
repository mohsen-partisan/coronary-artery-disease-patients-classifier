

from  DataPreprocessor import DataPreprocessor
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

class MeanClassFinder:

    def find_instances_near_two_class_centroid(self):
        data = DataPreprocessor().select_all_features()
        # data = data.drop(['target'], axis=1)

        mean_class_0 = data.loc[data['target'] == 0].mean()
        mean_class_1 = data.loc[data['target'] == 1].mean()
        means = pd.concat([mean_class_0, mean_class_1], axis=1)
        means = means.T
        data = data.append(means, ignore_index=True)
        class_0 = data.loc[data['target'] == 0]
        class_1 = data.loc[data['target'] == 1]



        dist_condensed_class_0 = pdist(class_0.values)
        distance_mat_0 = pd.DataFrame(squareform(dist_condensed_class_0), index=class_0.index,
                                    columns=class_0.index)
        distance_class_0 = distance_mat_0.iloc[(distance_mat_0.shape[0]-1)]
        distance_class_0.sort_values(ascending=True, inplace=True)
        distance_class_0.drop(labels=[distance_class_0.index[0]], inplace=True)
        sort_by_distance_class_0 = distance_class_0.index
        nearest_to_centroid_class_0 = sort_by_distance_class_0[0:len(sort_by_distance_class_0) // 2]
        not_near_to_centroid_0 = set(class_0.index) - set(nearest_to_centroid_class_0)


        dist_condensed_class_1 = pdist(class_1.values, metric='euclidean')
        distance_mat_1 = pd.DataFrame(squareform(dist_condensed_class_1), index=class_1.index,
                                      columns=class_1.index)
        distance_class_1 = distance_mat_1.iloc[(distance_mat_1.shape[0]-1)]
        distance_class_1.sort_values(ascending=True, inplace=True)
        distance_class_1.drop(labels=[distance_class_1.index[0]], inplace=True)
        sort_by_distance_class_1 = distance_class_1.index
        nearest_to_centroid_class_1 = sort_by_distance_class_1[0:len(sort_by_distance_class_1) // 2]
        not_near_to_centroid_1 = set(class_1.index) - set(nearest_to_centroid_class_1)


        new_data_class_0 = data.ix[not_near_to_centroid_0]
        new_data_class_1 = data.ix[nearest_to_centroid_class_1]
        useful_data = pd.concat([new_data_class_0, new_data_class_1])

        useful_data_index = np.concatenate([np.array(list(not_near_to_centroid_0)), nearest_to_centroid_class_1 ])
        remain_data_index = set(data.index) - set(useful_data_index)
        remain_data = data.ix[remain_data_index]

        return useful_data, remain_data


    def find_instances_near_three_class_centroid(self):
        data = DataPreprocessor().select_features()
        # data = data.drop(['target'], axis=1)

        mean_class_0 = data.loc[data['target'] == 0].mean()
        mean_class_1 = data.loc[data['target'] == 1].mean()
        mean_class_2 = data.loc[data['target'] == 2].mean()
        means = pd.concat([mean_class_0, mean_class_1, mean_class_2], axis=1)
        means = means.T
        data = data.append(means, ignore_index=True)
        class_0 = data.loc[data['target'] == 0]
        class_1 = data.loc[data['target'] == 1]
        class_2 = data.loc[data['target'] == 2]



        dist_condensed_class_0 = pdist(class_0.values)
        distance_mat_0 = pd.DataFrame(squareform(dist_condensed_class_0), index=class_0.index,
                                    columns=class_0.index)
        distance_class_0 = distance_mat_0.iloc[(distance_mat_0.shape[0]-1)]
        distance_class_0.sort_values(ascending=True, inplace=True)
        distance_class_0.drop(labels=[distance_class_0.index[0]], inplace=True)
        sort_by_distance_class_0 = distance_class_0.index
        nearest_to_centroid_class_0 = sort_by_distance_class_0[0:len(sort_by_distance_class_0) // 2]
        not_near_to_centroid_0 = set(class_0.index) - set(nearest_to_centroid_class_0)


        dist_condensed_class_1 = pdist(class_1.values, metric='euclidean')
        distance_mat_1 = pd.DataFrame(squareform(dist_condensed_class_1), index=class_1.index,
                                      columns=class_1.index)
        distance_class_1 = distance_mat_1.iloc[(distance_mat_1.shape[0]-1)]
        distance_class_1.sort_values(ascending=True, inplace=True)
        distance_class_1.drop(labels=[distance_class_1.index[0]], inplace=True)
        sort_by_distance_class_1 = distance_class_1.index
        nearest_to_centroid_class_1 = sort_by_distance_class_1[0:len(sort_by_distance_class_1) // 2]
        not_near_to_centroid_1 = set(class_1.index) - set(nearest_to_centroid_class_1)

        dist_condensed_class_2 = pdist(class_2.values, metric='euclidean')
        distance_mat_2 = pd.DataFrame(squareform(dist_condensed_class_2), index=class_2.index,
                                      columns=class_2.index)
        distance_class_2 = distance_mat_2.iloc[(distance_mat_2.shape[0] - 1)]
        distance_class_2.sort_values(ascending=True, inplace=True)
        distance_class_2.drop(labels=[distance_class_2.index[0]], inplace=True)
        sort_by_distance_class_2 = distance_class_2.index
        nearest_to_centroid_class_2 = sort_by_distance_class_2[0:len(sort_by_distance_class_2) // 2]
        not_near_to_centroid_2 = set(class_2.index) - set(nearest_to_centroid_class_2)


        new_data_class_0 = data.ix[not_near_to_centroid_0]
        new_data_class_1 = data.ix[not_near_to_centroid_1]
        new_data_class_2 = data.ix[nearest_to_centroid_class_2]
        useful_data = pd.concat([ new_data_class_0,new_data_class_1, new_data_class_2])

        useful_data_index = np.concatenate([np.array(list(not_near_to_centroid_0)) ,np.array(list(not_near_to_centroid_1)),
                                            nearest_to_centroid_class_2])
        remain_data_index = set(data.index) - set(useful_data_index)
        remain_data = data.ix[remain_data_index]

        return useful_data, remain_data






