-- Sum of northern latitudes between 38.7880 and 137.2345, truncated to 4 decimal places
SELECT TRUNCATE(SUM(LAT_N), 4) FROM STATION
WHERE LAT_N > 38.7880 AND LAT_N < 137.2345;
