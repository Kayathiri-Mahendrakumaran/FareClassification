from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import preprocess as pp
from sklearn.metrics import fbeta_score
from gsmote import oldgsmote
from gsmote import eg_smote
from imblearn.over_sampling import SMOTE


def xgboost(X_train, X_test, y_train, y_test):
    gbc = xgb.XGBClassifier(learning_rate=0.01, n_estimators=5000, max_depth=4, min_child_weight=6, gamma=0,
                            subsample=0.8, colsample_bylevel=0.8, reg_alpha=0.005, objective="binary:logistic",
                            nthread=4, scale_pos_weight=1, seed=27)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    f1score = fbeta_score(y_test, y_predict, average='weighted', beta=0.5)
    return ["xgboost", f1score]


def logistic_training(X_train, X_test, y_train, y_test):
    # Fitting Simple Linear Regression to the Training set
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = regressor.predict(X_test)
    y_predict = np.where(y_predict > 0.5, 1, 0)
    f1score = f1_score(y_test, y_predict, average='macro')
    return ["logistic regression", f1score]


def gradient_boosting(X_train, X_test, y_train, y_test):
    # Fitting Gradient boosting
    gbc = GradientBoostingClassifier(penalty="l2", C=1)
    gbc.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = gbc.predict(X_test)
    y_predict = np.where(y_predict.astype(int) > 0.5, 1, 0)

    f1score = f1_score(y_test, y_predict, average='macro')
    return ["gradient boosting", f1score]


def KNN(X_train, X_test, y_train, y_test):
    # Fitting Simple Linear Regression to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_predict = classifier.predict(X_test).astype(int)

    f1score = f1_score(y_test, y_predict, average='macro')
    return ["KNN", f1score]


train_filename = "Data/train.csv"
df_train = pd.read_csv(train_filename)
X, y = pp.preProcess_X(df_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# sm = oldgsmote()
# X_train, y_train = sm.fit_resample(X_train, y_train)

performance1 = xgboost(X_train, X_test, y_train, y_test)
# performance2 = logistic_training(X_train, X_test, y_train, y_test)
# performance3 = gradient_boosting(X_train, X_test, y_train, y_test)
# performance4 = KNN(X_train, X_test, y_train, y_test)

labels = ["Classifier", "f_score"]
values = [performance1,
          #           # performance2,
          #           # performance3,
          #           # performance4
          ]
scores = pd.DataFrame(values, columns=labels)
print(scores)
#


# Final evaluation
test_filename = "Data/test.csv"
df = pd.read_csv(test_filename)
X_t, y_t = pp.preProcess_X(df)
#
gbc = xgb.XGBClassifier(learning_rate=0.01, n_estimators=5000, max_depth=4, min_child_weight=6, gamma=0,
                        subsample=0.8, colsample_bylevel=0.8, reg_alpha=0.005, objective="binary:logistic",
                        nthread=4, scale_pos_weight=1, seed=27, penalty="l2")
gbc.fit(X, y)
#
#
# Predicting the Test set results
y_predict = gbc.predict(X_t)
data = np.column_stack([df.iloc[:, 0].values, y_predict])
label = ["tripid", "prediction"]
#
frame = pd.DataFrame(data, columns=label)
#
export_csv = frame.to_csv(r'output/prediction81.csv', index=None,
                          header=True)  # Don't forget to add '.csv' at the end of the path
#
