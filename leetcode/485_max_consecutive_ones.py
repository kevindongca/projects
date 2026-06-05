# LeetCode 485 - Max Consecutive Ones
# https://leetcode.com/problems/max-consecutive-ones/
# Difficulty: Easy
# Approach: Single pass counter, reset on 0, track running max
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Increment counter on 1, reset to 0 on anything else.
# Update high whenever counter exceeds it.
# Array is binary so high = 0 suffices as initial value.

from typing import List


class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        high = 0
        counter = 0
        for n in nums:
            if n == 1:
                counter += 1
            else:
                counter = 0
            if counter > high:
                high = counter
        return high
