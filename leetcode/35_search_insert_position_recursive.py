# LeetCode 35 - Search Insert Position
# https://leetcode.com/problems/search-insert-position/
# Difficulty: Easy
# Time Complexity: O(log n) | Space Complexity: O(log n) — call stack + slicing

from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if not nums:
            return 0
        m = len(nums) // 2
        if nums[m] < target:
            return m + 1 + self.searchInsert(nums[m+1:], target)
        elif nums[m] > target:
            return self.searchInsert(nums[:m], target)
        else:
            return m
