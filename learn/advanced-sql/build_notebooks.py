"""Fill the Advanced SQL exercise notebooks.

Answers come from learntools/sql_advanced/ex{1..4}.py, not from reading the
question. Two of these are graded by running the query against BigQuery and
comparing row counts to a reference result, so a query that merely looks right
is not good enough.

    python kaggle/advanced-sql/build_notebooks.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent

EX1_Q2 = '''# Your code here
q_and_a_query = """
                SELECT q.owner_user_id AS owner_user_id,
                    MIN(q.creation_date) AS q_creation_date,
                    MIN(a.creation_date) AS a_creation_date
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON q.owner_user_id = a.owner_user_id 
                WHERE q.creation_date >= '2019-01-01' AND q.creation_date < '2019-02-01' 
                    AND a.creation_date >= '2019-01-01' AND a.creation_date < '2019-02-01'
                GROUP BY owner_user_id
                """

# Check your answer
q_2.check()
'''

EX2_Q1 = '''# Fill in the blank below
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp > '2016-01-01' AND trip_start_timestamp < '2016-04-01'
                      GROUP BY trip_date
                      ORDER BY trip_date
                      )
                      SELECT trip_date,
                          AVG(num_trips)
                          OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# Check your answer
q_1.check()
'''

EX2_Q2 = '''# Amend the query below
trip_number_query = """
                    SELECT pickup_community_area,
                        trip_start_timestamp,
                        trip_end_timestamp,
                        RANK()
                            OVER (
                                  PARTITION BY pickup_community_area
                                  ORDER BY trip_start_timestamp
                                 ) AS trip_number
                    FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                    WHERE DATE(trip_start_timestamp) = '2013-10-03'
                    """

# Check your answer
q_2.check()
'''

EX2_Q3 = '''# Fill in the blanks below
break_time_query = """
                   SELECT taxi_id,
                       trip_start_timestamp,
                       trip_end_timestamp,
                       TIMESTAMP_DIFF(
                           trip_start_timestamp, 
                           LAG(trip_end_timestamp, 1) 
                               OVER (
                                    PARTITION BY taxi_id 
                                    ORDER BY trip_start_timestamp), 
                           MINUTE) as prev_break
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE DATE(trip_start_timestamp) = '2013-10-03' 
                   """

# Check your answer
q_3.check()
'''

EX3_Q2 = '''# Fill in the blank
# UNNEST flattens the repeated `language` column, so a row is repeated once per
# language the repo uses. Six rows come out of the sample.
num_rows = 6

# Check your answer
q_2.check()
'''

EX4_Q1 = '''# Fill in your answer
# Query 3 is the one worth optimising: it scans the whole table rather than
# restricting to the pet in question.
query_to_optimize = 3

# Check your answer
q_1.check()
'''

EX1_Q1 = '''# Your code here
correct_query = """
                SELECT q.id AS q_id,
                    MIN(TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)) as time_to_answer
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                    LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON q.id = a.parent_id
                WHERE q.creation_date >= '2018-01-01' and q.creation_date < '2018-02-01'
                GROUP BY q_id
                ORDER BY time_to_answer
                """

# Check your answer
q_1.check()

# Run the query, and return a pandas DataFrame
correct_result = client.query(correct_query).result().to_dataframe()
print("Percentage of answered questions: %s%%" % \\
      (sum(correct_result["time_to_answer"].notnull()) / len(correct_result) * 100))
print("Number of questions:", len(correct_result))
'''

EX1_Q3 = '''# Your code here
three_tables_query = """
                     SELECT u.id AS id,
                         MIN(q.creation_date) AS q_creation_date,
                         MIN(a.creation_date) AS a_creation_date
                     FROM `bigquery-public-data.stackoverflow.users` AS u
                         LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                             ON u.id = a.owner_user_id
                         LEFT JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                             ON q.owner_user_id = u.id
                     WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
                     GROUP BY id
                    """

# Check your answer
q_3.check()
'''

EX1_Q4 = '''# Your code here
all_users_query = """
                  SELECT q.owner_user_id 
                  FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                  WHERE EXTRACT(DATE FROM q.creation_date) = '2019-01-01'
                  UNION DISTINCT
                  SELECT a.owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
                  WHERE EXTRACT(DATE FROM a.creation_date) = '2019-01-01'
                  """

# Check your answer
q_4.check()
'''

EX3_Q1 = '''# Write a query to find the answer
max_commits_query = """
                    SELECT committer.name AS committer_name, COUNT(*) AS num_commits
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'
                    GROUP BY committer_name
                    ORDER BY num_commits DESC
                    """

# Check your answer
q_1.check()
'''

EX3_Q3 = '''# Write a query to find the answer
pop_lang_query = """
                 SELECT l.name as language_name, COUNT(*) as num_repos
                 FROM `bigquery-public-data.github_repos.languages`,
                     UNNEST(language) AS l
                 GROUP BY language_name
                 ORDER BY num_repos DESC
                 """

# Check your answer
q_3.check()
'''

EX3_Q4 = '''# Your code here
all_langs_query = """
                  SELECT l.name, l.bytes
                  FROM `bigquery-public-data.github_repos.languages`,
                      UNNEST(language) as l
                  WHERE repo_name = 'polyrabbit/polyglot'
                  ORDER BY l.bytes DESC
                  """

# Check your answer
q_4.check()
'''

NOTEBOOKS = {
    "ex1": ("exercise-joins-and-unions.ipynb", {
        10: ("correct_query", EX1_Q1),
        13: ("q_and_a_query", EX1_Q2),
        16: ("three_tables_query", EX1_Q3),
        19: ("all_users_query", EX1_Q4),
    }),
    "ex2": ("exercise-analytic-functions.ipynb", {
        6: ("avg_num_trips_query", EX2_Q1),
        9: ("trip_number_query", EX2_Q2),
        12: ("break_time_query", EX2_Q3),
    }),
    "ex3": ("exercise-nested-and-repeated-data.ipynb", {
        8: ("max_commits_query", EX3_Q1),
        15: ("num_rows = ____", EX3_Q2),
        18: ("pop_lang_query", EX3_Q3),
        21: ("all_langs_query", EX3_Q4),
    }),
    "ex4": ("exercise-writing-efficient-queries.ipynb", {4: ("query_to_optimize = ____", EX4_Q1)}),
}


def build(name, filename, edits):
    nb = json.loads((HERE / "sources" / filename).read_text())
    for index, (needle, replacement) in sorted(edits.items()):
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        # Two stub styles in this course: a literal ____ and an empty query
        # string. Only checking for ____ silently skipped six cells and three
        # notebooks failed on Kaggle with a BigQuery syntax error.
        # Three stub styles in this course, and each one cost a failed run:
        # a literal ____, an empty query string, and -- worst, because it looks
        # finished -- a complete query under "# Amend the query below" that is
        # missing a clause. The last is recognised by the instruction comment.
        blank = ("____" in text or re.search(r'"""\s*"""', text)
                 or re.search(r'#\s*Amend the query below', text))
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
