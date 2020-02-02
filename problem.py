import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit


problem_title = 'Immigration flows'
_target_column_name = 'Value'
Predictions = rw.prediction_types.make_regression()


class IMMI(FeatureExtractorRegressor):

    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'population.csv', 'taxes.csv',
            'governement_debt.csv', 'employment_rate.csv', 'greenhouse_gases.csv',
            'health.csv']):
        super(IMMI, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names


workflow = IMMI()


# define the score (specific score for the IMMI problem)
class IMMI_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='fan error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        max_true = np.log10(np.maximum(1., y_true))
        max_pred = np.log10(np.maximum(1., y_pred))
        loss = np.mean((max_true - max_pred)**2)
        return loss


score_types = [
    IMMI_error(name='fan error', precision=2),
]


def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y, X["Country"])


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,)
                       #compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[::30], y_array[::30]
    else:
        return X_df, y_array
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'inflows_train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'inflows_test.csv'
    return _read_data(path, f_name)
