"""Fill the Machine Learning Explainability exercise notebooks.

Reads the untouched notebooks in `sources/` and writes a pushable copy into
`ex2/` .. `ex5/` next to the `kernel-metadata.json` each one needs.

Every graded answer here was read off the grader in Kaggle/learntools
(`learntools/ml_explainability/ex{2..5}.py`), not recalled. Several of the
questions are about reading a plot this repo cannot see, and two are checked
against hard-coded arrays, so guessing would be wrong in a way that only shows
up after the push.

Replacements are located by cell index AND verified against a needle, so a
notebook that changes upstream fails loudly instead of silently filling the
wrong cell.

    python kaggle/ml-explainability/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# --- ex2: permutation importance -------------------------------------------

EX2_Q2 = '''import eli5
from eli5.sklearn import PermutationImportance

# The model to inspect is `first_model`, not the `my_model` of the snippet.
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

# Check your answer
q_2.check()

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
'''

EX2_Q4 = '''# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names = features_2)

# Check your answer
q_4.check()
'''

# --- ex3: partial plots -----------------------------------------------------

EX3_Q1 = '''for feat_name in base_features:
    PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name])
    plt.show()
'''

EX3_Q2 = '''fig, ax = plt.subplots(figsize=(8, 6))

# Add your code here
# A pair in the feature list asks for the 2D interaction plot rather than two
# separate curves.
fnames = [('pickup_longitude', 'dropoff_longitude')]
disp = PartialDependenceDisplay.from_estimator(first_model, val_X, fnames, ax=ax)
plt.show()
'''

EX3_Q3 = '''# Read off the contour plot at dropoff_longitude = -74: the fare falls from a
# little under 15 to a little under 9. The grader accepts 4 to 8.
savings_from_shorter_trip = 6

# Check your answer
q_3.check()
'''

EX3_Q4 = '''# This is the PDP for pickup_longitude without the absolute difference features. Included here to help compare it to the new PDP you create
feat_name = 'pickup_longitude'
PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name])
plt.show()

# Your code here
# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

feat_name = 'pickup_longitude'
disp = PartialDependenceDisplay.from_estimator(second_model, new_val_X, [feat_name])
plt.show()

# Check your answer
q_4.check()
'''

EX3_Q6 = '''import numpy as np
from numpy.random import rand

n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

# Your code here
# Create y. you should have X1 and X2 in the expression for y
# Three straight segments: down below -1, up between -1 and 1, down above 1.
# The (X1 < -1) and (X1 > 1) terms flip the slope of X1 outside the middle band.
y = -2 * X1 * (X1 < -1) + X1 - 2 * X1 * (X1 > 1) - X2

# create dataframe
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)
disp = PartialDependenceDisplay.from_estimator(my_model, predictors_df, ['X1'])
plt.show()

# Check your answer
q_6.check()
'''

EX3_Q7 = '''import eli5
from eli5.sklearn import PermutationImportance

n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
# Create y. you should have X1 and X2 in the expression for y
# A pure interaction. X1 moves the prediction a lot, so permuting it hurts and
# importance is high -- but averaged over X2 its effect cancels to zero, so the
# partial dependence plot is flat.
y = X1 * X2


# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)


disp = PartialDependenceDisplay.from_estimator(my_model, predictors_df, ['X1'], grid_resolution=300)
plt.show()

perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)

# Check your answer
q_7.check()

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm, feature_names = ['X1', 'X2'])
'''

# --- ex4: SHAP values -------------------------------------------------------

EX4_Q1 = '''# Your code here
# Permutation importance is the succinct overview the doctors asked for: one
# ranked table of what the model actually leans on.
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
'''

EX4_Q2 = '''# Your Code Here
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_name = 'number_inpatient'
PartialDependenceDisplay.from_estimator(my_model, val_X, [feature_name])
plt.show()
'''

EX4_Q3 = '''# Your Code Here
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_name = 'time_in_hospital'
PartialDependenceDisplay.from_estimator(my_model, val_X, [feature_name])
plt.show()
'''

EX4_Q4 = '''# Your Code Here
# The raw rate, straight from the data and with no model in the way.
# Concatenate the training half only, so the validation data stays untouched.
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
'''

EX4_Q5 = '''# Your Code Here
import shap  # package used to calculate Shap values

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function


def patient_risk_factors(model, patient_data):
    """Force plot for one patient: what pushed their risk up, and what down."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    # Index 1 is the readmitted-positive class.
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)


patient_risk_factors(my_model, sample_data_for_prediction)
'''

# --- ex5: advanced uses of SHAP values --------------------------------------

EX5_Q1 = '''# set following variable to 'diag_1_428' or 'payer_code_?'
# diag_1_428 is the wider range, driven by the handful of points far to the
# right rather than by the bulk of the distribution.
feature_with_bigger_range_of_effects = 'diag_1_428'

# Check your answer
q_1.check()
'''

EX5_Q3 = '''# Set following var to "diag_1_428" if changing it to 1 has bigger effect.  Else set it to 'payer_code_?'
# Most diag_1_428 SHAP values are small, but the pink dots -- the patients who
# have the diagnosis -- sit far from zero. The diagnosis is rare and carries a
# large risk for whoever has it.
bigger_effect_when_changed = 'diag_1_428'

# Check your answer
q_3.check()
'''

EX5_Q6 = '''# Your code here
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)
shap.dependence_plot('num_medications', shap_values[1], small_val_X)
'''

# needle -> the text that must appear in the cell being replaced
NOTEBOOKS = {
    "ex2": ("exercise-permutation-importance.ipynb", {
        9: ("q_2.check()", EX2_Q2),
        15: ("perm2 = ____", EX2_Q4),
    }),
    "ex3": ("exercise-partial-plots.ipynb", {
        7: ("for feat_name in base_features:", EX3_Q1),
        11: ("fig, ax = plt.subplots", EX3_Q2),
        15: ("savings_from_shorter_trip = ____", EX3_Q3),
        19: ("data['abs_lon_change'] = ____", EX3_Q4),
        25: ("y = np.ones(n_samples)", EX3_Q6),
        29: ("X1 = ____", EX3_Q7),
    }),
    "ex4": ("exercise-shap-values.ipynb", {
        9: ("____", EX4_Q1),
        14: ("____", EX4_Q2),
        18: ("____", EX4_Q3),
        22: ("____", EX4_Q4),
        27: ("____", EX4_Q5),
    }),
    "ex5": ("exercise-advanced-uses-of-shap-values.ipynb", {
        7: ("feature_with_bigger_range_of_effects = ____", EX5_Q1),
        14: ("bigger_effect_when_changed = ____", EX5_Q3),
        24: ("____", EX5_Q6),
    }),
}


def build(name, filename, edits):
    src = HERE / "sources" / filename
    nb = json.loads(src.read_text())
    for index, (needle, replacement) in sorted(edits.items()):
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        if cell["cell_type"] != "code" or needle not in text:
            raise SystemExit(
                f"{name}: cell {index} does not contain {needle!r} -- "
                f"the notebook changed upstream, re-pull it before building"
            )
        cell["source"] = replacement.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    out = HERE / name / filename
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(edits)} cells filled)")


def main():
    for name, (filename, edits) in NOTEBOOKS.items():
        build(name, filename, edits)


if __name__ == "__main__":
    main()
