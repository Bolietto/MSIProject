import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import DistanceMetric

class implementedKNORAE(BaseEstimator, ClassifierMixin):
    """
    Our very own fantastic and outperforming version of KNORA-E.

    Parameters
    ----------
     pool_classifiers : list of classifiers

     k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base
        classifiers.

    
    """

    

    def __init__(self, pool_classifiers = None, k = 7):
        self.pool_classifiers = pool_classifiers
        self.k = k


    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y
       

        return self

    def get_competence_region(self, x_query, Validation_set):
        competence_region = []

        for val_point in Validation_set:
            dist=DistanceMetric.get_metric('euclidean')
            points=[x_query, val_point[0]]
            result=np.array(dist.pairwise(points)).max()
            competence_region.append([val_point, result])

        competence_region = sorted(competence_region, key=lambda x: x[1])[:self.k]
        competence_region = [c[0] for c in competence_region]

        return competence_region

    def validate_region(self, Classifier, competence_region):
        for point in competence_region:
            arr = point[0].reshape(1, -1)
            prediction = Classifier.predict(arr)
            if prediction != point[1]:
                return False
        return True

    def reduce_region(self, region_of_competence):
        region_of_competence = region_of_competence[:-1]
        return region_of_competence

    def predict(self, X):
        y_pred = []

        for x_query in X:
            competence_region = self.get_competence_region(x_query, zip(self.X_, self.y_))
            ensemble_of_classifiers = []

            while (len(ensemble_of_classifiers) == 0) and (len(competence_region) > 0):
                  for c in self.pool_classifiers:
                      if self.validate_region(c, competence_region):
                          ensemble_of_classifiers.append(c)
                  if len(ensemble_of_classifiers) == 0:
                      competence_region = self.reduce_region(competence_region)

            if len(ensemble_of_classifiers) == 0:
                ensemble_of_classifiers = self.pool_classifiers

            predicton = 0
            for Clf in ensemble_of_classifiers:
                p = Clf.predict(x_query.reshape(1, -1))
                predicton += p

            if predicton > len(ensemble_of_classifiers)/2:
                y_pred.append(1)
            elif predicton < len(ensemble_of_classifiers)/2:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return y_pred
