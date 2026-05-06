-- Cities that do not start or do not end with a vowel, no duplicates
SELECT DISTINCT CITY FROM STATION
WHERE LEFT(CITY, 1) NOT IN ('A', 'E', 'I', 'O', 'U')
OR RIGHT(CITY, 1) NOT IN ('A', 'E', 'I', 'O', 'U');
