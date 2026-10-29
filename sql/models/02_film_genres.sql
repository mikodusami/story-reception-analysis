-- One row per (film, genre).
CREATE OR REPLACE TABLE film_genres AS
SELECT movie_url, trim(genre) AS genre
FROM films, UNNEST(genres) AS t(genre);
