-- One row per audience review. reviewer_id is a hash of the display name (first name + initial).
CREATE OR REPLACE TABLE audience_reviews AS
SELECT
    movie_url,
    reviewer_id,
    CAST(score AS DOUBLE)                AS rating_0_5,       -- 0.5 to 5 stars
    CAST(rating_normalized AS DOUBLE)    AS rating_0_1,
    CAST(verified AS BOOLEAN)            AS verified,
    language,
    CAST(word_count AS INTEGER)          AS word_count,
    review_text
FROM stg_audience_reviews;
