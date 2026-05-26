# LeetCode 704 - Binary Search
# https://leetcode.com/problems/binary-search/
# Difficulty: Easy
# Approach: Iterative binary search
# Time Complexity: O(log n) | Space Complexity: O(1)

from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while high >= low:
            mid = (high + low) // 2
            if target == nums[mid]:
                return mid
            if target < nums[mid]:
                high = mid - 1
            if target > nums[mid]:
                low = mid + 1
        return -1
