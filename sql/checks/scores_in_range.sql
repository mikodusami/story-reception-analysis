-- Scores stay on their scales.
SELECT 'films.tomatometer' AS col, count(*) AS n FROM films WHERE tomatometer NOT BETWEEN 0 AND 100 HAVING count(*) > 0
UNION ALL SELECT 'films.audience_score', count(*) FROM films WHERE audience_score NOT BETWEEN 0 AND 100 HAVING count(*) > 0
UNION ALL SELECT 'books.avg_rating', count(*) FROM books WHERE avg_rating NOT BETWEEN 1 AND 5 HAVING count(*) > 0
UNION ALL SELECT 'critic_reviews.rating_0_5', count(*) FROM critic_reviews WHERE rating_0_5 NOT BETWEEN 0 AND 5 HAVING count(*) > 0
UNION ALL SELECT 'audience_reviews.rating_0_5', count(*) FROM audience_reviews WHERE rating_0_5 NOT BETWEEN 0.5 AND 5 HAVING count(*) > 0
UNION ALL SELECT 'book_reviews.stars', count(*) FROM book_reviews WHERE stars NOT BETWEEN 1 AND 5 HAVING count(*) > 0
UNION ALL
SELECT 'rating_0_1 (all reviews)', count(*) FROM (
    SELECT rating_0_1 FROM critic_reviews UNION ALL SELECT rating_0_1 FROM audience_reviews
    UNION ALL SELECT rating_0_1 FROM book_reviews
) WHERE rating_0_1 NOT BETWEEN 0 AND 1 HAVING count(*) > 0;
