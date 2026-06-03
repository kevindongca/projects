# LeetCode 414 - Third Maximum Number
# https://leetcode.com/problems/third-maximum-number/
# Difficulty: Easy
# Approach: Convert to set (remove duplicates), sort descending, index directly
# Time Complexity: O(n log n) | Space Complexity: O(n)
#
# set() removes duplicates in O(n) — handles the "distinct" requirement cleanly.
# After sorting descending, index [2] is the 3rd max if it exists, else [0] is the max.
#
# Previous approach used sequential list comprehension filtering (O(n^2) due to
# calling max() inside the comprehension each iteration). This is strictly better.

from typing import List


class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        nums = list(set(nums))
        nums.sort(reverse=True)
        if len(nums) >= 3:
            return nums[2]
        else:
            return nums[0]
