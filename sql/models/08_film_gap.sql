-- One row per film with both scores: how far critics and audiences disagree.
CREATE OR REPLACE TABLE film_gap AS
SELECT
    f.movie_url,
    f.title,
    f.release_year,
    f.content_rating,
    f.runtime_min,
    f.tomatometer,
    f.audience_score,
    f.tomatometer - f.audience_score                  AS gap,          -- positive: critics liked it more
    abs(f.tomatometer - f.audience_score)             AS abs_gap,
    CASE WHEN f.tomatometer - f.audience_score >= 20  THEN 'critics 20+ higher'
         WHEN f.tomatometer - f.audience_score <= -20 THEN 'audiences 20+ higher'
         ELSE 'within 20 points' END                  AS split,
    (SELECT count(*) FROM critic_reviews c WHERE c.movie_url = f.movie_url)    AS critic_reviews_scraped,
    (SELECT count(*) FROM audience_reviews a WHERE a.movie_url = f.movie_url)  AS audience_reviews_scraped
FROM films f
WHERE f.tomatometer IS NOT NULL AND f.audience_score IS NOT NULL;
