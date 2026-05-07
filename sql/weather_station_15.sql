-- Western longitude corresponding to the greatest northern latitude under 137.2345
SELECT ROUND(LONG_W, 4) FROM STATION
WHERE LAT_N = (SELECT MAX(LAT_N) FROM STATION WHERE LAT_N < 137.2345);
