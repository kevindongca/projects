# LeetCode 268 - Missing Number
# https://leetcode.com/problems/missing-number/
# Difficulty: Easy
# Approach (initial): Sort, scan for index != value
# Time Complexity: O(n log n) | Space Complexity: O(1)

from typing import List


class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums)):
            if i != nums[i]:
                return i
        return i + 1
