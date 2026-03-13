import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading train data and test data, converting to dataframes
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)

# converting DateTime from text into useful numeric columns for train and test
train["DateTime"] = pd.to_datetime(train["DateTime"], errors="coerce")
train["hour"] = train["DateTime"].dt.hour

test["DateTime"] = pd.to_datetime(test["DateTime"], errors="coerce")
test["hour"] = test["DateTime"].dt.hour


# removing the id column and DateTIme column (and response variable from test data)
train = train.drop(columns = ["id", "DateTime"])
test  = test.drop(columns  = ["id", "DateTime", "meal"], errors = "ignore")

# setting up the target and features
y = train["meal"]
X = train.drop(columns = ["meal"])

# splitting to test accuracy
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    test_size = 0.2, 
    random_state = 1
)
test  = test[X.columns]

# XGBoost model
model = XGBClassifier(
    n_estimators = 300,
    max_depth = 4,
    learning_rate = 0.1,
    random_state = 1
)

# fitting model on training split
modelFit = model.fit(X_train, y_train)

# checking accuracy
valid_pred = modelFit.predict(X_valid)
accuracy = accuracy_score(y_valid, valid_pred)
print("Model Accuracy:", accuracy)

# refitting model on all training data
modelFit = model.fit(X, y)

# saving fitted model
joblib.dump(modelFit, "xgboost_model.pkl")

# had error on validPred test - hoping this solves that issue
pred = [int(x) for x in modelFit.predict(test)]
print("Number of predictions:", len(pred))
