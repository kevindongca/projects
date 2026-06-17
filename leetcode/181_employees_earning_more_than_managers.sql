-- LeetCode 181 - Employees Earning More Than Their Managers
-- https://leetcode.com/problems/employees-earning-more-than-their-managers/
-- Difficulty: Easy
-- Approach: Self-join Employee table as employee and manager, filter by salary
--
-- Join Employee to itself: e is the employee, m is their manager.
-- ON e.managerId = m.id links each employee to their manager row.
-- WHERE e.salary > m.salary filters to employees out-earning their manager.

SELECT e.name AS Employee
FROM Employee e
JOIN Employee m ON e.managerId = m.id
WHERE e.salary > m.salary;
