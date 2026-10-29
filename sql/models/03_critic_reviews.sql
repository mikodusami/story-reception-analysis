-- One row per critic review.
CREATE OR REPLACE TABLE critic_reviews AS
SELECT
    movie_url,
    critic_name,
    publication,
    CAST(is_positive AS BOOLEAN)         AS is_fresh,
    CAST(rating_parsed AS DOUBLE)        AS rating_0_5,       -- NULL when the critic gave no numeric rating
    CAST(rating_normalized AS DOUBLE)    AS rating_0_1,
    language,
    CAST(word_count AS INTEGER)          AS word_count,
    review_text
FROM stg_critic_reviews;
