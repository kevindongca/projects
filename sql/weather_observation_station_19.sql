-- HackerRank: Weather Observation Station 19
-- Category: Aggregation
-- https://www.hackerrank.com/challenges/weather-observation-station-19
-- Euclidean distance between P1(min_lat, min_long) and P2(max_lat, max_long)

SELECT ROUND(SQRT(POW(MAX(LONG_W) - MIN(LONG_W), 2) + POW(MAX(LAT_N) - MIN(LAT_N), 2)), 4)
FROM STATION;
