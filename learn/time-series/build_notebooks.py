"""Fill the Time Series exercise notebooks.

Answers come from learntools/time_series/ex{1..6}.py. Most are checked by
comparing the produced Series or DataFrame against a reference computed inside
the grader, so the values have to match, not just the shape of the code.

Two places deviate from the grader's own solution text, both deliberate and
both noted at the call site: `OneHotEncoder(sparse=...)` was renamed in
scikit-learn 1.2 and removed in 1.4, and the grader's snippet still uses the
removed spelling.

    python kaggle/time-series/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# --- ex1: linear regression with time series --------------------------------

EX1_Q3 = '''from sklearn.linear_model import LinearRegression

df = average_sales.to_frame()

# YOUR CODE HERE: Create a time dummy
# A plain 0..n-1 counter. Regressing on it fits a straight line in time.
time = np.arange(len(df.index))

df['time'] = time 

# YOUR CODE HERE: Create training data
X = df.loc[:, ['time']]  # features
y = df.loc[:, 'sales']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)


# Check your answer
q_3.check()
'''

EX1_Q4 = '''df = average_sales.to_frame()

# YOUR CODE HERE: Create a lag feature from the target 'sales'
# shift(1) moves each value one step forward, so each row sees yesterday.
lag_1 = df['sales'].shift(1)

df['lag_1'] = lag_1  # add to dataframe

X = df.loc[:, ['lag_1']].dropna()  # features
y = df.loc[:, 'sales']  # target
y, X = y.align(X, join='inner')  # drop corresponding values in target

# YOUR CODE HERE: Create a LinearRegression instance and fit it to X and y.
model = LinearRegression()
model.fit(X, y)

# YOUR CODE HERE: Create Store the fitted values as a time series with
# the same time index as the training data
y_pred = pd.Series(model.predict(X), index=X.index)


# Check your answer
q_4.check()
'''

EX2_Q1 = '''# YOUR CODE HERE: Add methods to `food_sales` to compute a moving
# average with appropriate parameters for trend estimation.
# Twelve months is one full seasonal cycle, so a 12-month window averages the
# season away and leaves the trend. center=True keeps it from lagging, and
# min_periods=6 lets the curve start before a whole window is available.
trend = food_sales.rolling(
    window=12,
    center=True,
    min_periods=6,
).mean()

# Check your answer
q_1.check()

# Make a plot
ax = food_sales.plot(**plot_params, alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)
'''

# --- ex2: trend -------------------------------------------------------------

EX2_Q3 = '''from statsmodels.tsa.deterministic import DeterministicProcess

y = average_sales.copy()  # the target

# YOUR CODE HERE: Instantiate `DeterministicProcess` with arguments
# appropriate for a cubic trend model
# Cubic means order=3: the process builds t, t**2 and t**3 for us.
dp = DeterministicProcess(index=y.index, order=3)

# YOUR CODE HERE: Create the feature set for the dates given in y.index
X = dp.in_sample()

# YOUR CODE HERE: Create features for a 90-day forecast.
X_fore = dp.out_of_sample(steps=90)


# Check your answer
q_3.check()
'''

# --- ex3: seasonality -------------------------------------------------------

EX3_Q2 = '''y = average_sales.copy()

# YOUR CODE HERE
# Four Fourier pairs give a smooth monthly shape without one dummy per day.
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    # YOUR CODE HERE
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()

# Check your answer
q_2.check()
'''

EX3_Q4 = '''# YOUR CODE HERE
from sklearn.preprocessing import OneHotEncoder

# scikit-learn renamed `sparse` to `sparse_output` in 1.2 and removed the old
# name in 1.4. The course's own solution still uses the removed spelling, so
# try the current one first.
try:
    ohe = OneHotEncoder(sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(sparse=False)

X_holidays = pd.DataFrame(
    ohe.fit_transform(holidays),
    index=holidays.index,
    columns=holidays.description.unique(),
)

X2 = X.join(X_holidays, on='date').fillna(0.0)

# Check your answer
q_4.check()
'''

# --- ex4: time series as features -------------------------------------------

EX4_Q1 = '''# YOUR CODE HERE
# center=True puts the window around each point, so the average does not lag.
y_ma = y.rolling(7, center=True).mean()


# Plot
ax = y_ma.plot()
ax.set_title("Seven-Day Moving Average");

# Check your answer
q_1.check()
'''

EX4_Q4 = '''# YOUR CODE HERE: Make features from `y_deseason`
X_lags = make_lags(y_deseason, lags=1)

# YOUR CODE HERE: Make features from `onpromotion`
# You may want to use `pd.concat`
# Promotions are known in advance, so the model may look one step ahead as well
# as behind -- that is what make_leads is for.
X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    make_leads(onpromotion, leads=1),
], axis=1)

X = pd.concat([X_time, X_lags, X_promo], axis=1).dropna()
y, X = y.align(X, join='inner')

# Check your answer
q_4.check()
'''

EX4_Q5 = '''y_lag = supply_sales.loc[:, 'sales'].shift(1)
onpromo = supply_sales.loc[:, 'onpromotion']

# 28-day mean of lagged target
mean_7 = y_lag.rolling(7).mean()
# YOUR CODE HERE: 14-day median of lagged target
median_14 = y_lag.rolling(14).median()
# YOUR CODE HERE: 7-day rolling standard deviation of lagged target
std_7 = y_lag.rolling(7).std()
# YOUR CODE HERE: 7-day sum of promotions with centered window
promo_7 = onpromo.rolling(7, center=True).sum()


# Check your answer
q_5.check()
'''

# --- ex5: hybrid models -----------------------------------------------------

EX5_Q1 = '''def fit(self, X_1, X_2, y):
    # YOUR CODE HERE: fit self.model_1
    self.model_1.fit(X_1, y)

    y_fit = pd.DataFrame(
        # YOUR CODE HERE: make predictions with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )

    # YOUR CODE HERE: compute residuals
    # What model_1 failed to explain is exactly what model_2 gets to learn.
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() # wide to long

    # YOUR CODE HERE: fit self.model_2 on residuals
    self.model_2.fit(X_2, y_resid)

    # Save column names for predict method
    self.y_columns = y.columns
    # Save data for question checking
    self.y_fit = y_fit
    self.y_resid = y_resid


# Add method to class
BoostedHybrid.fit = fit


# Check your answer
q_1.check()
'''

EX5_Q2 = '''def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
        # YOUR CODE HERE: predict with self.model_1
        self.model_1.predict(X_1),
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # YOUR CODE HERE: add self.model_2 predictions to y_pred
    # model_2 predicts the residual, so the two predictions add.
    y_pred += self.model_2.predict(X_2)
    
    return y_pred.unstack()  # long to wide


# Add method to class
BoostedHybrid.predict = predict


# Check your answer
q_2.check()
'''

EX5_Q3 = '''# YOUR CODE HERE: Create LinearRegression + XGBRegressor hybrid with BoostedHybrid
# Linear model takes the trend, the booster takes what is left.
model = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=XGBRegressor(),
)

# YOUR CODE HERE: Fit and predict
model.fit(X_1, X_2, y)
y_pred = model.predict(X_1, X_2)

y_pred = y_pred.clip(0.0)


# Check your answer
q_3.check()
'''

# --- ex6: forecasting with machine learning ---------------------------------

EX6_Q1 = '''# YOUR CODE HERE: Match the task to the dataset. Answer 1, 2, or 3.
task_a = 2
task_b = 1
task_c = 3

# Check your answer
q_1.check()
'''

EX6_Q3 = '''# YOUR CODE HERE
y = family_sales.loc[:, 'sales']

# YOUR CODE HERE: Make 4 lag features
X = make_lags(y, lags=4).dropna()

# YOUR CODE HERE: Make multistep target
# 16 steps out, one column per horizon: the direct multi-output strategy.
y = make_multistep_target(y, steps=16).dropna()

y, X = y.align(X, join='inner', axis=0)

# Check your answer
q_3.check()
'''

EX6_Q4 = '''from sklearn.multioutput import RegressorChain

# YOUR CODE HERE
# A chain feeds each step's prediction into the next, so later horizons see
# what the earlier ones forecast.
model = RegressorChain(base_estimator=XGBRegressor())

# Check your answer
q_4.check()
'''

NOTEBOOKS = {
    "ex1": ("exercise-linear-regression-with-time-series.ipynb",
            {14: ("time = ____", EX1_Q3), 19: ("lag_1 = ____", EX1_Q4)}),
    "ex2": ("exercise-trend.ipynb",
            {8: ("trend = food_sales", EX2_Q1), 15: ("dp = ____", EX2_Q3)}),
    "ex3": ("exercise-seasonality.ipynb",
            {10: ("fourier = ____", EX3_Q2), 24: ("X_holidays = ____", EX3_Q4)}),
    "ex4": ("exercise-time-series-as-features.ipynb",
            {6: ("y_ma = ____", EX4_Q1), 17: ("X_lags = ____", EX4_Q4),
             23: ("median_14 = ____", EX4_Q5)}),
    "ex5": ("exercise-hybrid-models.ipynb",
            {6: ("def fit(self, X_1, X_2, y):", EX5_Q1),
             9: ("def predict(self, X_1, X_2):", EX5_Q2),
             14: ("model = ____", EX5_Q3)}),
    "ex6": ("exercise-forecasting-with-machine-learning.ipynb",
            {6: ("task_a = ____", EX6_Q1), 13: ("X = ____", EX6_Q3),
             18: ("model = ____", EX6_Q4)}),
}


def build(name, filename, edits):
    nb = json.loads((HERE / "sources" / filename).read_text())
    for index, (needle, replacement) in sorted(edits.items()):
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        # Not every stub uses ____. The trend exercise ships `trend = food_sales`,
        # a bare assignment that runs fine and answers nothing.
        blank = "____" in text or "trend = food_sales\n" in text
        if cell["cell_type"] != "code" or needle not in text or not blank:
            raise SystemExit(f"{name}: cell {index} is not the stub for {needle!r}")
        cell["source"] = replacement.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    out = HERE / name / filename
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(edits)} cells)")


def main():
    for name, (filename, edits) in NOTEBOOKS.items():
        build(name, filename, edits)


if __name__ == "__main__":
    main()
