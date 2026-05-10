-- HackerRank: Placements
-- Category: Advanced Join
-- https://www.hackerrank.com/challenges/placements
-- Find students whose best friends got a higher salary offer than them
-- Order by the friend's salary

SELECT STUDENTS.NAME
FROM STUDENTS
JOIN FRIENDS ON STUDENTS.ID = FRIENDS.ID
JOIN PACKAGES P1 ON STUDENTS.ID = P1.ID
JOIN PACKAGES P2 ON FRIENDS.FRIEND_ID = P2.ID
WHERE P2.SALARY > P1.SALARY
ORDER BY P2.SALARY;
