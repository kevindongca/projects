# LeetCode 263 - Ugly Number
# https://leetcode.com/problems/ugly-number/
# Difficulty: Easy
# Approach: Repeatedly divide out factors of 2, 3, 5 — check if 1 remains
# Time Complexity: O(log n) | Space Complexity: O(1)
#
# An ugly number's only prime factors are 2, 3, and 5.
# Divide out all occurrences of each. If n reduces to 1, it's ugly.
# If anything else remains, it has another prime factor -> not ugly.
# Edge case: n <= 0 is never ugly by definition.

class Solution:
    def isUgly(self, n: int) -> bool:
        if n <= 0:
            return False
        for i in [2, 3, 5]:
            while n % i == 0:
                n //= i
        return n == 1
