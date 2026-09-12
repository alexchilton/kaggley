"""Fill the Intermediate Machine Learning exercise notebooks.

Answers from learntools/ml_intermediate/ex{1..7}.py. Each cell is written out
whole rather than patched, because most of these stubs have several blanks in
one cell and a per-blank substitution would be harder to read than the result.

One deviation from a published solution: the categorical-variables exercise
uses `OneHotEncoder(sparse=False)`, renamed to `sparse_output` in
scikit-learn 1.2 and removed in 1.4. Handled the same way as the Time Series
course -- try the current name, fall back to the old one.

    python kaggle/intermediate-ml/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

CELLS = {
"exercise-introduction.ipynb": {
 12: '''# Fill in the best model
# model_3 has the lowest MAE of the five.
best_model = model_3

# Check your answer
step_1.check()
''',
 15: '''# Define a model
my_model = best_model # Your code here

# Check your answer
step_2.check()
''',
},
"exercise-missing-values.ipynb": {
 10: '''# Fill in the line below: How many rows are in the training data?
num_rows = 1168

# Fill in the line below: How many columns in the training data
# have missing values?
num_cols_with_missing = 3

# Fill in the line below: How many missing entries are contained in
# all of the training data?
tot_missing = 212 + 6 + 58

# Check your answer
step_1.a.check()
''',
 18: '''# Fill in the line below: get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Check your answer
step_2.check()
''',
 23: '''from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Check your answer
step_3.a.check()
''',
 31: '''# Preprocessed training and validation features
# The median is less sensitive to the outliers in these columns than the mean.
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Imputation removed column names; put them back
final_X_train.columns = X_train.columns
final_X_valid.columns = X_valid.columns

# Check your answer
step_4.a.check()
''',
 36: '''# Fill in the line below: preprocess test data
# transform, not fit_transform: the test data must not inform the imputer.
final_X_test = pd.DataFrame(final_imputer.transform(X_test))
final_X_test.columns = X_test.columns

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

# Check your answer
step_4.b.check()
''',
},
"exercise-categorical-variables.ipynb": {
 10: '''# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# Check your answer
step_1.check()
''',
 22: '''from sklearn.preprocessing import OrdinalEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

# Check your answer
step_2.b.check()
''',
 29: '''# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25

# Check your answer
step_3.a.check()
''',
 32: '''# Fill in the line below: How many entries are added to the dataset by 
# replacing the column with a one-hot encoding?
# 10,000 rows x 100 new columns, minus the single original column.
OH_entries_added = 1e4*100 - 1e4

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with an ordinal encoding?
# An ordinal encoding replaces the column in place, so nothing is added.
label_entries_added = 0

# Check your answer
step_3.b.check()
''',
 37: '''from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
# scikit-learn renamed `sparse` to `sparse_output` in 1.2 and removed the old
# name in 1.4; the course's solution still uses the removed spelling.
try:
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# Check your answer
step_4.check()
''',
},
"exercise-pipelines.ipynb": {
 9: '''# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant') # Your code here

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Check your answer
step_1.a.check()
''',
 15: '''# Preprocessing of test data, fit model
# The pipeline applies the fitted preprocessing itself, so the raw test frame
# goes straight in.
preds_test = my_pipeline.predict(X_test) # Your code here

# Check your answer
step_2.check()
''',
},
"exercise-cross-validation.ipynb": {
 12: '''def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    # cross_val_score returns the negated MAE, so flip the sign back.
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

# Check your answer
step_1.check()
''',
 15: '''results = {50*i: get_score(50*i) for i in range(1,9)} # Your code here

# Check your answer
step_2.check()
''',
 20: '''# The key with the lowest average MAE.
n_estimators_best = min(results, key=results.get)

# Check your answer
step_3.check()
''',
},
"exercise-xgboost.ipynb": {
 6: '''from xgboost import XGBRegressor

# Define the model
my_model_1 = XGBRegressor(random_state=0) # Your code here

# Fit the model
my_model_1.fit(X_train, y_train) # Your code here

# Check your answer
step_1.a.check()
''',
 9: '''from sklearn.metrics import mean_absolute_error

# Get predictions
predictions_1 = my_model_1.predict(X_valid) # Your code here

# Check your answer
step_1.b.check()
''',
 12: '''# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)

# Check your answer
step_1.c.check()
''',
 15: '''# Define the model
# More trees at a smaller learning rate: slower to fit, lower error.
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05) # Your code here

# Fit the model
my_model_2.fit(X_train, y_train) # Your code here

# Get predictions
predictions_2 = my_model_2.predict(X_valid) # Your code here

# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)

# Check your answer
step_2.check()
''',
 18: '''# Define the model
# One tree, on purpose: this question wants a model that is clearly worse.
my_model_3 = XGBRegressor(n_estimators=1)

# Fit the model
my_model_3.fit(X_train, y_train) # Your code here

# Get predictions
predictions_3 = my_model_3.predict(X_valid)

# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)

# Check your answer
step_3.check()
''',
},
"exercise-data-leakage.ipynb": {
 12: '''# Fill in the line below with one of 1, 2, 3 or 4.
# The neighbourhood average sale price is updated after a sale completes, so
# the target of the row being predicted can feed back into its own feature.
potential_leakage_feature = 2

# Check your answer
q_5.check()
''',
},
}

FILES = {
 "ex1": "exercise-introduction.ipynb",
 "ex2": "exercise-missing-values.ipynb",
 "ex3": "exercise-categorical-variables.ipynb",
 "ex4": "exercise-pipelines.ipynb",
 "ex5": "exercise-cross-validation.ipynb",
 "ex6": "exercise-xgboost.ipynb",
 "ex7": "exercise-data-leakage.ipynb",
}


def main():
    for name, filename in FILES.items():
        nb = json.loads((HERE / "sources" / filename).read_text())
        edits = CELLS.get(filename, {})
        for index, new in sorted(edits.items()):
            cell = nb["cells"][index]
            text = "".join(cell["source"])
            # Not every stub is ____. The cross-validation exercise ships a
            # function whose body is `pass`, which runs fine and answers
            # nothing.
            if "____" not in text and "\n    pass" not in text:
                raise SystemExit(f"{filename}: cell {index} has no stub to fill")
            cell["source"] = new.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        out = HERE / name / filename
        out.write_text(json.dumps(nb, indent=1) + "\n")
        left = sum(1 for c in nb["cells"]
                   if c["cell_type"] == "code" and "____" in "".join(c["source"]))
        print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(edits)} cells, {left} ____ left)")


if __name__ == "__main__":
    main()
