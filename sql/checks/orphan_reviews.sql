-- Every review belongs to a film or book that exists.
SELECT 'critic_reviews' AS tbl, count(*) AS n FROM critic_reviews c
WHERE NOT EXISTS (SELECT 1 FROM films f WHERE f.movie_url = c.movie_url) HAVING count(*) > 0
UNION ALL
SELECT 'audience_reviews', count(*) FROM audience_reviews a
WHERE NOT EXISTS (SELECT 1 FROM films f WHERE f.movie_url = a.movie_url) HAVING count(*) > 0
UNION ALL
SELECT 'book_reviews', count(*) FROM book_reviews r
WHERE NOT EXISTS (SELECT 1 FROM books b WHERE b.book_id = r.book_id) HAVING count(*) > 0;
