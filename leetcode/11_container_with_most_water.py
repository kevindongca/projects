# LeetCode 11 - Container With Most Water
# https://leetcode.com/problems/container-with-most-water/
# Difficulty: Medium
# Approach: Two pointers from both ends, always move the shorter wall inward
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Why move the shorter wall:
# The water height is capped by the SHORTER of the two walls. Moving the taller
# wall inward keeps the same cap but loses width -> can never improve. So the
# only move with upside is to discard the shorter wall and look for a taller one.
# When the walls are equal, move either side (here: left); the container is
# already recorded and any future container using that wall is strictly narrower.

from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        best = 0
        left = 0
        right = len(height) - 1
        while left < right:
            area = (right - left) * min(height[left], height[right])
            best = max(area, best)
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return best
