-- Difference between actual and miscalculated average salary (zeros removed), rounded up
SELECT CEIL(AVG(SALARY) - AVG(REPLACE(SALARY, '0', ''))) FROM EMPLOYEES;
