-- HackerRank: Weather Observation Station 20
-- Category: Aggregation
-- https://www.hackerrank.com/challenges/weather-observation-station-20
-- Median of LAT_N rounded to 4 decimal places

-- Approach 1: LIMIT/OFFSET (logical approach, blocked by HackerRank MySQL runtime)
-- SELECT ROUND(LAT_N, 4)
-- FROM STATION
-- ORDER BY LAT_N
-- LIMIT 1 OFFSET (SELECT FLOOR(COUNT(*) / 2) FROM STATION);

-- Approach 2: Correlated subquery (accepted solution)
-- Count how many LAT_N values are less than the current row.
-- The median row is the one where that count equals FLOOR((N-1)/2).
SELECT ROUND(S.LAT_N, 4)
FROM STATION S
WHERE (
    SELECT COUNT(LAT_N)
    FROM STATION
    WHERE LAT_N < S.LAT_N
) = (
    SELECT FLOOR((COUNT(*) - 1) / 2)
    FROM STATION
);
