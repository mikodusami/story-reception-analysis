-- Load the cleaned CSVs written by data-cleaning.ipynb. {{clean_dir}} is filled in by the runner.
CREATE OR REPLACE TABLE stg_movies AS
SELECT * FROM read_csv('{{clean_dir}}/cleaned_rotten_tomatoes_movies.csv', header = true);

CREATE OR REPLACE TABLE stg_critic_reviews AS
SELECT * FROM read_csv('{{clean_dir}}/cleaned_rotten_tomatoes_critic_reviews.csv', header = true);

CREATE OR REPLACE TABLE stg_audience_reviews AS
SELECT * FROM read_csv('{{clean_dir}}/cleaned_rotten_tomatoes_audience_reviews.csv', header = true);

CREATE OR REPLACE TABLE stg_books AS
SELECT * FROM read_csv('{{clean_dir}}/cleaned_goodreads_books.csv', header = true);

CREATE OR REPLACE TABLE stg_book_reviews AS
SELECT * FROM read_csv('{{clean_dir}}/cleaned_goodreads_reviews.csv', header = true);
