-- One row per film genre with 30+ films: average critic, audience and gap, and how often it splits.
-- Simple averages (a film with several genres counts in each); main.ipynb has the adjusted Ridge estimates.
CREATE OR REPLACE TABLE genre_gap AS
SELECT
    g.genre,
    count(*)                                                  AS films,
    round(avg(fg.tomatometer), 1)                             AS avg_tomatometer,
    round(avg(fg.audience_score), 1)                          AS avg_audience_score,
    round(avg(fg.gap), 1)                                     AS avg_gap,
    round(avg((fg.split = 'critics 20+ higher')::INTEGER), 3)   AS share_critics_20_higher,
    round(avg((fg.split = 'audiences 20+ higher')::INTEGER), 3) AS share_audiences_20_higher
FROM film_gap fg
JOIN film_genres g USING (movie_url)
GROUP BY g.genre
HAVING count(*) >= 30
ORDER BY avg_gap DESC;
