# LeetCode 70 - Climbing Stairs
# https://leetcode.com/problems/climbing-stairs/
# Difficulty: Easy
# Approach: Recursive Fibonacci
# Time Complexity: O(2^n) | Space Complexity: O(n)
# Note: TLEs on LeetCode due to exponential time — repeated subproblems
# Fix: use iterative DP (see 70_climbing_stairs_iterative.py)


class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        return self.climbStairs(n-1) + self.climbStairs(n-2)
