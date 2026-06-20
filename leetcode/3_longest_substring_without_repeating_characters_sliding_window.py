# LeetCode 3 - Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Difficulty: Medium
# Approach (OFFICIAL): Sliding window with set
# Time Complexity: O(n) | Space Complexity: O(min(n, 26))
#
# a = left pointer, window = set of chars in current window.
# Expand right with for loop. When duplicate found, shrink left until gone.
# Update max_length every iteration: window size = b - a + 1.
# No edge case handling needed — empty string returns 0 naturally.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        a = 0
        window = set()
        max_length = 0
        for b in range(len(s)):
            while s[b] in window:
                window.remove(s[a])
                a += 1
            window.add(s[b])
            max_length = max(max_length, b - a + 1)
        return max_length
