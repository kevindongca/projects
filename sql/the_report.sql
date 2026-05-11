-- HackerRank: The Report
-- Category: Basic Join
-- https://www.hackerrank.com/challenges/the-report
-- Generate a report with Name, Grade, and Marks
-- Students with grade < 8 show NULL as name
-- Order by grade DESC, name ASC for grade 8-10, marks ASC for grade 1-7

SELECT CASE WHEN GRADES.GRADE < 8 THEN NULL ELSE NAME END, GRADES.GRADE, MARKS
FROM STUDENTS
JOIN GRADES ON STUDENTS.MARKS BETWEEN MIN_MARK AND MAX_MARK
ORDER BY GRADES.GRADE DESC,
         CASE WHEN GRADES.GRADE >= 8 THEN NAME END ASC,
         MARKS ASC;
