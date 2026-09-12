# Advanced SQL

The four exercises of the [Advanced SQL](https://www.kaggle.com/learn/advanced-sql)
course: JOINs and UNIONs, analytic functions, nested and repeated data, and
writing efficient queries.

| Dir | Exercise | Stubs filled |
| --- | --- | --- |
| `ex1/` | JOINs and UNIONs | 1 |
| `ex2/` | Analytic Functions | 2 |
| `ex3/` | Nested and Repeated Data | 1 |
| `ex4/` | Writing Efficient Queries | 1 |

## Where the answers come from

`learntools/sql_advanced/ex{1..4}.py`. Two of these graders actually run the
query against BigQuery and compare the row count and a specific value to a
reference result, so a query that reads correctly but returns the wrong rows
still fails. They cannot be checked locally -- there is no BigQuery here -- so
the run on Kaggle is the only verification.

The answers themselves are small: a `FULL JOIN`, a windowed `AVG` over
`ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING`, a `LAG` partitioned by taxi,
`num_rows = 6`, and `query_to_optimize = 3`.
