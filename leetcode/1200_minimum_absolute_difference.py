# LeetCode 1200 - Minimum Absolute Difference
# https://leetcode.com/problems/minimum-absolute-difference/
# Difficulty: Easy
# Approach: Sort, then slide adjacent pairs tracking minimum diff seen so far
# Time Complexity: O(n log n) | Space Complexity: O(n)
#
# Key insight: minimum absolute difference always occurs between sorted neighbours.
# If two elements aren't adjacent after sorting, something smaller lies between them.
# So sort first, then one pass of adjacent comparisons is sufficient.

from typing import List


class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        output = []
        curr = float('inf')
        left, right = 0, 1
        arr.sort()
        while right != len(arr):
            diff = arr[right] - arr[left]
            if diff < curr:
                output = [[arr[left], arr[right]]]
                curr = diff
            elif diff == curr:
                output.append([arr[left], arr[right]])
            left += 1
            right += 1
        return output
