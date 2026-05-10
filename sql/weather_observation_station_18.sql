-- HackerRank: Weather Observation Station 18
-- Category: Aggregation
-- https://www.hackerrank.com/challenges/weather-observation-station-18
-- Manhattan distance between P1(min_lat, min_long) and P2(max_lat, max_long)

SELECT ROUND(ABS(MAX(LAT_N) - MIN(LAT_N)) + ABS(MAX(LONG_W) - MIN(LONG_W)), 4)
FROM STATION;
