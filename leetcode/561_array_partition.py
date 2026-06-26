# LeetCode 561 - Array Partition
# https://leetcode.com/problems/array-partition/
# Difficulty: Easy
# Approach: Sort, sum every element at even indices (min of each pair)
# Time Complexity: O(n log n) | Space Complexity: O(1)
#
# After sorting, optimal pairing is adjacent elements: (nums[0],nums[1]), (nums[2],nums[3])...
# min of each pair is always the left element (even index) after sorting.
# sum of nums[0], nums[2], nums[4]... gives maximum possible sum of minimums.

from typing import List


class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[i] for i in range(0, len(nums), 2))
