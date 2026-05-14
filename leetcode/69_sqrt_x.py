# LeetCode 69 - Sqrt(x)
# https://leetcode.com/problems/sqrtx/
# Difficulty: Easy
# Time Complexity: O(sqrt(n)) | Space Complexity: O(1)
# Note: optimal approach is binary search O(log n)


class Solution:
    def mySqrt(self, x: int) -> int:
        n = 1
        while n * n <= x:
            n += 1
        return n - 1
