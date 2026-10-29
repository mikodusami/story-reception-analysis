-- One row per (book, genre tag).
CREATE OR REPLACE TABLE book_genres AS
SELECT book_id, trim(genre) AS genre
FROM books, UNNEST(genres) AS t(genre);
