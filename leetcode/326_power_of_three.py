# LeetCode 326 - Power of Three
# https://leetcode.com/problems/power-of-three/
# Difficulty: Easy
# Approach: Iterative — count up powers of 3 until overshoot, check last hit
# Time Complexity: O(log n) | Space Complexity: O(1)
#
# O(1) alternative: return n > 0 and 3**19 % n == 0
# 3**19 = 1162261467 is the largest power of 3 in 32-bit int range.
# Since 3 is prime, any power of 3 must divide 3**19 evenly.

class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        i = 0
        while 3**i <= n:
            i += 1
        return 3**(i-1) == n
