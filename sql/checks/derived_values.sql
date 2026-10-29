-- Derived columns agree with their inputs.
SELECT 'film_gap.gap' AS col, count(*) AS n FROM film_gap WHERE gap <> tomatometer - audience_score HAVING count(*) > 0
UNION ALL
SELECT 'books.star_sd within 0-2', count(*) FROM books WHERE star_sd NOT BETWEEN 0 AND 2 HAVING count(*) > 0
UNION ALL
-- the star counts should reproduce Goodreads' published average (allowing for its rounding)
SELECT 'books star mean vs avg_rating', count(*) FROM (
    SELECT (1*coalesce(rating_dist_1,0) + 2*coalesce(rating_dist_2,0) + 3*coalesce(rating_dist_3,0)
            + 4*coalesce(rating_dist_4,0) + 5*coalesce(rating_dist_5,0))
           / nullif(coalesce(rating_dist_1,0) + coalesce(rating_dist_2,0) + coalesce(rating_dist_3,0)
                    + coalesce(rating_dist_4,0) + coalesce(rating_dist_5,0), 0) AS star_mean,
           avg_rating
    FROM stg_books
) WHERE abs(star_mean - avg_rating) > 0.05 HAVING count(*) > 0;
