"""Fill the Feature Engineering exercise notebooks.

Answers from learntools/feature_engineering_new/. Note the module numbers are
offset by one from the notebook order -- exercise-mutual-information imports
ex2, not ex1 -- so the solutions have to be matched by reading each notebook's
import rather than by position. Taking the obvious mapping would have pulled
the wrong answers into four of the five notebooks.

    mutual-information            -> ex2   (all ThoughtExperiment)
    creating-features             -> ex3   (q_1..q_5 CodingProblem)
    clustering-with-k-means       -> ex4
    principal-component-analysis  -> ex5
    target-encoding               -> ex6

Two questions are scored rather than compared. PCA q_2 asserts the score beats
0.140 RMSLE, and target-encoding q_2 rebuilds the encoding itself and compares
frames, so the encoder has to match on columns and m.

    python kaggle/feature-engineering/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

CELLS = {
"exercise-creating-features.ipynb": {
 4: '''# YOUR CODE HERE
X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch


# Check your answer
q_1.check()
''',
 7: '''# YOUR CODE HERE
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
# Multiply
# Each dummy column becomes "living area, but only for this building type".
X_2 = X_2.mul(df.GrLivArea, axis=0)


# Check your answer
q_2.check()
''',
 10: '''X_3 = pd.DataFrame()

# YOUR CODE HERE
# Count how many of the porch types this house actually has.
X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)


# Check your answer
q_3.check()
''',
 15: '''X_4 = pd.DataFrame()

# YOUR CODE HERE
# MSSubClass values look like "One_Story_1946_and_Newer_All_Styles"; the first
# underscore-separated token is the general class.
X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

# Check your answer
q_4.check()
''',
 18: '''X_5 = pd.DataFrame()

# YOUR CODE HERE
# transform keeps the original row index, so the neighbourhood median lines up
# with each house rather than collapsing the frame.
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

# Check your answer
q_5.check()
''',
},
"exercise-clustering-with-k-means.ipynb": {
 6: '''X = df.copy()
y = X.pop("SalePrice")


# YOUR CODE HERE: Define a list of the features to be used for the clustering
features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]


# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)


# YOUR CODE HERE: Fit the KMeans model to X_scaled and create the cluster labels
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)


# Check your answer
q_2.check()
''',
 13: '''kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)


# YOUR CODE HERE: Create the cluster-distance features using `fit_transform`
# fit_transform returns the distance to every centroid, not the label: ten
# columns of "how far is this house from cluster i".
X_cd = kmeans.fit_transform(X_scaled)


# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)


# Check your answer
q_3.check()
''',
},
"exercise-principal-component-analysis.ipynb": {
 11: '''X = df.copy()
y = X.pop("SalePrice")

# YOUR CODE HERE: Add new features to X.
# Suggested by the loadings: size above and below ground together, and
# remodel date weighted by basement size. The grader wants RMSLE < 0.140.
X["Feature1"] = X.GrLivArea + X.TotalBsmtSF
X["Feature2"] = X.YearRemodAdd * X.TotalBsmtSF

score = score_dataset(X, y)
print(f"Your score: {score:.5f} RMSLE")


# Check your answer
q_2.check()
''',
},
"exercise-target-encoding.ipynb": {
 12: '''# YOUR CODE HERE: Create the MEstimateEncoder
# Choose a set of features to encode and a value for m
# Neighborhood has the most categories and many rare ones, which is exactly
# where target encoding helps; m=1.0 is light smoothing.
encoder = MEstimateEncoder(
    cols=["Neighborhood"],
    m=1.0,
)


# Fit the encoder on the encoding split
encoder.fit(X_encode, y_encode)

# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)


# Check your answer
q_2.check()
''',
},
}

FILES = {
 "ex1": "exercise-mutual-information.ipynb",
 "ex2": "exercise-creating-features.ipynb",
 "ex3": "exercise-clustering-with-k-means.ipynb",
 "ex4": "exercise-principal-component-analysis.ipynb",
 "ex5": "exercise-target-encoding.ipynb",
}


def main():
    for name, filename in FILES.items():
        nb = json.loads((HERE / "sources" / filename).read_text())
        edits = CELLS.get(filename, {})
        for index, new in sorted(edits.items()):
            cell = nb["cells"][index]
            if "____" not in "".join(cell["source"]):
                raise SystemExit(f"{filename}: cell {index} has no ____ to fill")
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
