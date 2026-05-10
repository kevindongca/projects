# LeetCode 35 - Search Insert Position
# https://leetcode.com/problems/search-insert-position/
# Difficulty: Easy
# Time Complexity: O(log n) | Space Complexity: O(1)

from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            m = (left + right) // 2
            if nums[m] == target:
                return m
            elif nums[m] < target:
                left = m + 1
            elif nums[m] > target:
                right = m - 1
        return left
