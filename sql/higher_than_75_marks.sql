-- Students with marks above 75, ordered by last 3 characters of name then ID
SELECT NAME FROM STUDENTS
WHERE MARKS > 75
ORDER BY RIGHT(NAME, 3), ID;
