-- Cities starting and ending with a vowel, no duplicates
SELECT DISTINCT CITY FROM STATION
WHERE LEFT(CITY, 1) IN ('A', 'E', 'I', 'O', 'U')
AND RIGHT(CITY, 1) IN ('A', 'E', 'I', 'O', 'U');
