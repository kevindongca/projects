# LeetCode 231 - Power of Two
# https://leetcode.com/problems/power-of-two/
# Difficulty: Easy
# Approach: Iterative division by 2 until reaching 2 or 1
# Time Complexity: O(log n) | Space Complexity: O(1)
#
# O(1) bit trick alternative: return n > 0 and n & (n-1) == 0
# Powers of 2 have exactly one 1-bit. n-1 flips that bit and sets all lower bits.
# n & (n-1) == 0 only when n has a single 1-bit i.e. is a power of 2.
# e.g. 8 = 1000, 7 = 0111, 8 & 7 = 0000 -> True
#      6 = 0110, 5 = 0101, 6 & 5 = 0100 -> False

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 1:
            return True
        while n / 2 >= 2:
            n = n / 2
        return n == 2

# Cleaner O(1):
# def isPowerOfTwo(self, n: int) -> bool:
#     return n > 0 and n & (n - 1) == 0
