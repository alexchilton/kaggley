# Time Series

The six exercises of the [Time Series](https://www.kaggle.com/learn/time-series)
course. Lessons only -- the Store Sales competition is not submitted to here.

| Dir | Exercise | Stubs filled |
| --- | --- | --- |
| `ex1/` | Linear Regression with Time Series | 2 |
| `ex2/` | Trend | 1 |
| `ex3/` | Seasonality | 2 |
| `ex4/` | Time Series as Features | 3 |
| `ex5/` | Hybrid Models | 3 |
| `ex6/` | Forecasting With Machine Learning | 3 |

`sources/` holds the notebooks exactly as pulled from Kaggle;
`build_notebooks.py` writes the filled copies next to each
`kernel-metadata.json`.

## Where the answers come from

`learntools/time_series/ex{1..6}.py`. Most questions are `EqualityCheckProblem`
against a Series or DataFrame the grader computes itself, so the values have to
match -- code that merely looks correct fails.

## One deliberate deviation

The seasonality exercise's holiday question uses `OneHotEncoder(sparse=False)`
in the course's own solution. scikit-learn renamed that argument to
`sparse_output` in 1.2 and removed the old spelling in 1.4, so the published
solution raises `TypeError` on a current image. The notebook here tries the
current name and falls back to the old one. The grader compares the resulting
frame, not the code, so the fallback does not affect the mark.

`CalendarFourier(freq='M', ...)` is kept as the course writes it. pandas has
deprecated `'M'` in favour of `'ME'`, so this may warn; it is left alone unless
a run actually fails on it.
