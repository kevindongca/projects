-- Query 1: Name followed by first letter of occupation in parentheses
SELECT CONCAT(NAME, '(', LEFT(OCCUPATION, 1), ')')
FROM OCCUPATIONS
ORDER BY NAME;

-- Query 2: Count of each occupation in sentence format
SELECT CONCAT('There are a total of ', COUNT(OCCUPATION), ' ', LOWER(OCCUPATION), 's.')
FROM OCCUPATIONS
GROUP BY OCCUPATION
ORDER BY COUNT(OCCUPATION), OCCUPATION;
