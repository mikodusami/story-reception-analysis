-- One row per Goodreads book, with how divided its readers are.
-- star_sd is the standard deviation of the book's 1-5 star ratings (higher = more divisive).
CREATE OR REPLACE TABLE books AS
WITH dist AS (
    SELECT
        *,
        coalesce(rating_dist_1, 0) AS d1, coalesce(rating_dist_2, 0) AS d2, coalesce(rating_dist_3, 0) AS d3,
        coalesce(rating_dist_4, 0) AS d4, coalesce(rating_dist_5, 0) AS d5
    FROM stg_books
), moments AS (
    SELECT
        *,
        d1 + d2 + d3 + d4 + d5                                        AS star_count,
        (1*d1 + 2*d2 + 3*d3 + 4*d4 + 5*d5) / nullif(d1 + d2 + d3 + d4 + d5, 0) AS star_mean
    FROM dist
)
SELECT
    book_id,
    title,
    author,
    CAST(publication_year AS INTEGER)    AS publication_year,
    CAST(avg_rating AS DOUBLE)           AS avg_rating,
    CAST(ratings_count AS BIGINT)        AS ratings_count,
    CAST(star_count AS BIGINT)           AS star_count,
    sqrt((d1 * (1 - star_mean)^2 + d2 * (2 - star_mean)^2 + d3 * (3 - star_mean)^2
          + d4 * (4 - star_mean)^2 + d5 * (5 - star_mean)^2) / nullif(star_count, 0)) AS star_sd,
    (d1 + d2) / nullif(star_count, 0)    AS share_1_2_stars,
    d5 / nullif(star_count, 0)           AS share_5_stars,
    list_filter(string_split(coalesce(genres, ''), '|'), lambda g: trim(g) <> '') AS genres
FROM moments;
