-- Cities starting with a vowel, no duplicates
SELECT DISTINCT CITY FROM STATION WHERE LEFT(CITY, 1) IN ('A', 'E', 'I', 'O', 'U');
