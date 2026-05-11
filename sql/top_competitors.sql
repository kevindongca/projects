-- HackerRank: Top Competitors
-- Category: Basic Join
-- https://www.hackerrank.com/challenges/full-score
-- Find hackers who achieved full scores on more than one challenge
-- Order by number of full scores DESC, hacker_id ASC

SELECT Hackers.hacker_id, Hackers.name
FROM Submissions
JOIN Challenges ON Submissions.challenge_id = Challenges.challenge_id
JOIN Difficulty ON Challenges.difficulty_level = Difficulty.difficulty_level
JOIN Hackers ON Submissions.hacker_id = Hackers.hacker_id
WHERE Submissions.score = Difficulty.score
GROUP BY Hackers.hacker_id, Hackers.name
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC, Hackers.hacker_id ASC;
