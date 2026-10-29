-- One row per Rotten Tomatoes film.
CREATE OR REPLACE TABLE films AS
SELECT
    movie_url,
    title,
    CAST(release_year AS INTEGER)            AS release_year,
    CAST(runtime_min AS INTEGER)             AS runtime_min,
    content_rating,
    original_language,
    CAST(tomatometer AS DOUBLE)              AS tomatometer,          -- % of critics positive, 0-100
    CAST(tomatometer_count AS INTEGER)       AS critic_count,
    CAST(critic_avg_rating AS DOUBLE)        AS critic_avg_rating,    -- out of 10
    CAST(top_critic_score AS DOUBLE)         AS top_critic_score,
    CAST(audience_score AS DOUBLE)           AS audience_score,       -- % of audience 3.5+ stars, 0-100
    CAST(audience_avg_rating AS DOUBLE)      AS audience_avg_rating,  -- out of 5
    CAST(audience_review_count AS INTEGER)   AS audience_count,
    list_filter(string_split(coalesce(genres, ''), '|'), lambda g: trim(g) <> '') AS genres
FROM stg_movies;
