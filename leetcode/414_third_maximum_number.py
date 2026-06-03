# LeetCode 414 - Third Maximum Number
# https://leetcode.com/problems/third-maximum-number/
# Difficulty: Easy
# Approach: Sequential filtering via list comprehensions to find distinct top 3
# Time Complexity: O(n) | Space Complexity: O(n)
#
# Filter out the max to get candidates for 2nd max, filter again for 3rd max.
# If fewer than 3 distinct values exist (empty list after filtering), return overall max.
# Note: max(second) returns the 3rd distinct maximum since first and second
# maximums have already been excluded.

from typing import List


class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        first = [item for item in nums if item != max(nums)]
        if first == []:
            return max(nums)
        second = [item for item in first if item != max(first)]
        if second == []:
            return max(nums)
        return max(second)
