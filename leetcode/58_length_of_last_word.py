# LeetCode 58 - Length of Last Word
# https://leetcode.com/problems/length-of-last-word/
# Difficulty: Easy
# Time Complexity: O(n) | Space Complexity: O(n)


class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.strip().split()[-1])
