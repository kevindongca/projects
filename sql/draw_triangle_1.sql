-- Draw triangle pattern P(20) using REPEAT and session variable
SET @r = 0;
SELECT REPEAT('* ', @r := @r + 1) FROM information_schema.tables LIMIT 20;
