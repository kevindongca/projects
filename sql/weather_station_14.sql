-- Greatest northern latitude less than 137.2345, truncated to 4 decimal places
SELECT TRUNCATE(MAX(LAT_N), 4) FROM STATION
WHERE LAT_N < 137.2345;
