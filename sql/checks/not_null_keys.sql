-- No missing keys.
SELECT 'films.movie_url' AS key, count(*) AS n FROM films WHERE movie_url IS NULL HAVING count(*) > 0
UNION ALL SELECT 'books.book_id', count(*) FROM books WHERE book_id IS NULL HAVING count(*) > 0
UNION ALL SELECT 'critic_reviews.movie_url', count(*) FROM critic_reviews WHERE movie_url IS NULL HAVING count(*) > 0
UNION ALL SELECT 'audience_reviews.movie_url', count(*) FROM audience_reviews WHERE movie_url IS NULL HAVING count(*) > 0
UNION ALL SELECT 'book_reviews.book_id', count(*) FROM book_reviews WHERE book_id IS NULL HAVING count(*) > 0;
