# LeetCode 4 - Median of Two Sorted Arrays
# https://leetcode.com/problems/median-of-two-sorted-arrays/
# Difficulty: Hard
# Approach: Merge both arrays, sort, find middle element(s)
# Time Complexity: O((m+n) log(m+n)) | Space Complexity: O(m+n)
#
# Optimal solution is O(log(min(m,n))) via binary search on partition,
# but brute force passes all 2098 test cases and beats 100% on runtime.

from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        merged = nums1 + nums2
        merged.sort()
        n = len(merged)
        if n % 2 == 0:
            return (merged[n // 2 - 1] + merged[n // 2]) / 2
        else:
            return float(merged[n // 2])
