# LeetCode 1295 - Find Numbers with Even Number of Digits
# https://leetcode.com/problems/find-numbers-with-even-number-of-digits/
# Difficulty: Easy
# Approach: Convert each number to string, check if digit count is even
# Time Complexity: O(n) | Space Complexity: O(1)
#
# len(str(n)) gives the digit count. Check parity with % 2 == 0.
# sum() with a generator expression counts matches without a separate counter.
#
# Original approach (also correct):
# for i in range(len(nums)):
#     if len(str(nums[i])) % 2 == 0: count += 1

from typing import List


class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return sum(1 for n in nums if len(str(n)) % 2 == 0)
