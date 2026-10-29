-- Each film, book and review appears once.
SELECT 'films.movie_url' AS key, movie_url AS value, count(*) AS n FROM films GROUP BY movie_url HAVING count(*) > 1
UNION ALL
SELECT 'books.book_id', CAST(book_id AS VARCHAR), count(*) FROM books GROUP BY book_id HAVING count(*) > 1
UNION ALL
SELECT 'book_reviews (book_id, reviewer_id)', book_id || ' / ' || reviewer_id, count(*)
FROM book_reviews GROUP BY book_id, reviewer_id HAVING count(*) > 1;
