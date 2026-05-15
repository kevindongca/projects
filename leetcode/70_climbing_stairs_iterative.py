# LeetCode 70 - Climbing Stairs
# https://leetcode.com/problems/climbing-stairs/
# Difficulty: Easy
# Approach: Iterative Fibonacci (Dynamic Programming)
# Time Complexity: O(n) | Space Complexity: O(1)

import math


class Solution:
    def climbStairs(self, n: int) -> int:
        x = 1
        y = 2
        temp = 0
        for i in range(n-1):
            temp = y
            y = y + x
            x = temp
        return x
