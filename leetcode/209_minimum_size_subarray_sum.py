# LeetCode 209 - Minimum Size Subarray Sum
# https://leetcode.com/problems/minimum-size-subarray-sum/
# Difficulty: Medium
# Approach: Sliding window — expand right, shrink left while sum >= target
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Classic variable-size sliding window template:
# Expand right unconditionally, shrink left when window is valid (sum >= target).
# While shrinking, record the window size and subtract the left element.
# minsub initialized to inf; if never updated, no valid subarray exists -> return 0.

import math
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        minsub = math.inf
        curr_sum = 0
        for right in range(len(nums)):
            curr_sum += nums[right]
            while curr_sum >= target:
                minsub = min(minsub, right - left + 1)
                curr_sum -= nums[left]
                left += 1
        return minsub if minsub != math.inf else 0
