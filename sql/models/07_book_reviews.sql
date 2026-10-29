-- One row per Goodreads review.
CREATE OR REPLACE TABLE book_reviews AS
SELECT
    book_id,
    reviewer_id,
    CAST(rating AS INTEGER)              AS stars,            -- NULL when the reader left text but no stars
    CAST(rating_normalized AS DOUBLE)    AS rating_0_1,
    language,
    CAST(word_count AS INTEGER)          AS word_count,
    review_text
FROM stg_book_reviews;
