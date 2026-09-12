"""Fill the Data Visualization exercise notebooks.

Answers from learntools/data_viz_to_coder/ex{1..7}.

Most questions are plots checked by inspecting the current matplotlib figure,
so the call has to produce the right kind of chart from the right columns. A
few are data lookups compared by value (`high_score = 7.759930`,
`worst_genre = 'Simulation'`), read out of the grader rather than recomputed.

ex6 has a single CodingProblem with no blanks -- run it for credit. ex7 is the
final project and is handled separately, see FINAL_PROJECT below.

    python kaggle/data-visualization/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def stub(text, replacement):
    return text.replace("____", replacement, 1)


# Each entry is cell index -> the text that replaces the single `____`.
INLINE = {
"exercise-hello-seaborn.ipynb": {
 5: "1",
},
"exercise-line-charts.ipynb": {
 6: 'pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)',
 9: "museum_data.tail()",
 14: "sns.lineplot(data=museum_data)",
 17: "sns.lineplot(data=museum_data['Avila Adobe'])",
},
"exercise-bar-charts-and-heatmaps.ipynb": {
 6: 'pd.read_csv(ign_filepath, index_col="Platform")',
 9: "ign_data",
 14: "sns.barplot(x=ign_data['Racing'], y=ign_data.index)",
 20: "sns.heatmap(ign_data, annot=True)",
},
"exercise-scatter-plots.ipynb": {
 6: 'pd.read_csv(candy_filepath, index_col="id")',
 9: "candy_data.head()",
 14: "sns.scatterplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])",
 20: "sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])",
 26: "sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'], hue=candy_data['chocolate'])",
 29: 'sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)',
 35: "sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent'])",
},
"exercise-distributions.ipynb": {
 7: 'pd.read_csv(cancer_filepath, index_col="Id")',
 10: "cancer_data.head()",
 16: "sns.histplot(data=cancer_data, x='Area (mean)', hue='Diagnosis')",
 22: "sns.kdeplot(data=cancer_data, x='Radius (worst)', hue='Diagnosis', shade=True)",
},
}

# Cells with more than one blank are written out whole.
FULL = {
"exercise-line-charts.ipynb": {
 11: '''# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = 2620

# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = 14658

# Check your answers
step_2.check()
''',
},
"exercise-bar-charts-and-heatmaps.ipynb": {
 11: '''# Fill in the line below: What is the highest average score received by PC games,
# for any genre?
high_score = 7.759930

# Fill in the line below: On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'

# Check your answers
step_2.check()
''',
},
"exercise-scatter-plots.ipynb": {
 11: '''# Fill in the line below: Which candy was more popular with survey respondents:
# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)
more_popular = '3 Musketeers'

# Fill in the line below: Which candy has higher sugar content: 'Air Heads'
# or 'Baby Ruth'? (Please enclose your answer in single quotes.)
more_sugar = 'Air Heads'

# Check your answers
step_2.check()
''',
},
"exercise-distributions.ipynb": {
 12: '''# Fill in the line below: In the first five rows of the data, what is the
# largest value for 'Perimeter (mean)'?
max_perim = 87.46

# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 8510824?
mean_radius = 9.504

# Check your answers
step_2.check()
''',
},
}

# The final project asks you to bring your own dataset. Nothing is attached to
# the notebook, so step_1 (`len(os.listdir('../input')) > 0` plus a .csv) fails
# outright. Pointing it at a dataset the course already ships keeps every check
# honest -- a real CSV, really loaded, really plotted -- without inventing data.
FINAL_PROJECT = {
 8: '''# Fill in the line below: Specify the path of the CSV file to read
# Bringing your own dataset is the point of this exercise; this uses one of the
# course's own files so the notebook is runnable end to end.
my_filepath = "../input/spotify.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
''',
 10: '''# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath, index_col="Date", parse_dates=True)

# Check that a dataset has been uploaded into my_data
step_3.check()
''',
 14: '''# Create a plot
plt.figure(figsize=(14, 6))
plt.title("Daily global streams, by song")
sns.lineplot(data=my_data)
plt.xlabel("Date")

# Check that a figure appears below
step_4.check()
''',
}

FILES = {
 "ex1": "exercise-hello-seaborn.ipynb",
 "ex2": "exercise-line-charts.ipynb",
 "ex3": "exercise-bar-charts-and-heatmaps.ipynb",
 "ex4": "exercise-scatter-plots.ipynb",
 "ex5": "exercise-distributions.ipynb",
 "ex6": "exercise-choosing-plot-types-and-custom-styles.ipynb",
 "ex7": "exercise-final-project.ipynb",
}


def main():
    for name, filename in FILES.items():
        nb = json.loads((HERE / "sources" / filename).read_text())
        inline = INLINE.get(filename, {})
        full = dict(FULL.get(filename, {}))
        if filename == "exercise-final-project.ipynb":
            full.update(FINAL_PROJECT)

        for index in sorted(set(inline) | set(full)):
            cell = nb["cells"][index]
            text = "".join(cell["source"])
            if "____" not in text:
                raise SystemExit(f"{filename}: cell {index} has no ____ to fill")
            new = full[index] if index in full else stub(text, inline[index])
            cell["source"] = new.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None

        out = HERE / name / filename
        out.write_text(json.dumps(nb, indent=1) + "\n")
        left = sum(1 for c in nb["cells"]
                   if c["cell_type"] == "code" and "____" in "".join(c["source"]))
        print(f"wrote {out.relative_to(HERE.parents[1])}  "
              f"({len(inline) + len(full)} cells, {left} ____ left)")


if __name__ == "__main__":
    main()
