import xgboost as xgb
import numpy as np
import pandas as pd
import preprocess as pp
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from gsmote.oldgsmote import OldGeometricSMOTE
# from gsmote.eg_smote import EGSmote
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE

train_filename = "Data/train.csv"
df_train = pd.read_csv(train_filename)
X = pp.preProcess_X(df_train)
y = pp.preProcess_y(df_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
# sm = EGSmote()
# X_train, y_train = sm.fit_resample(X_train, y_train)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

params = {
        'min_child_weight': [1, 2, 5],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7]
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
# sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
# sm = EGSmote()
# X, y = sm.fit_resample(X, y)
folds = 3
param_comb = 600

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='f1_macro', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, y)
timer(start_time) # timing ends here for "start_time" variable

# print('\n All results:')
# print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)