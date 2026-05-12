# LeetCode 28 - Find the Index of the First Occurrence in a String
# https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/
# Difficulty: Easy
# Time Complexity: O(n*m) | Space Complexity: O(1)


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # Note: built-in solution. Manual implementation would use sliding window or KMP.
        return haystack.find(needle)
