-- Sum of all latitude and longitude values rounded to 2 decimal places
SELECT ROUND(SUM(LAT_N), 2), ROUND(SUM(LONG_W), 2) FROM STATION;
