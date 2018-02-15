# -*- coding:utf-8 -*-
"""
resample dataset to solve imbalance problem
"""
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def smote_tomek_oversampling(X, y):
    smt = SMOTETomek(random_state=2,
                     smote=SMOTE(kind='svm')
                     )
    X_resampled, y_resampled = smt.fit_sample(X, y)
    return X_resampled, y_resampled
