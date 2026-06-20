# LeetCode 3 - Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Difficulty: Medium
# Approach (INITIAL): Manual two-pointer with set, if/elif branch structure
# Time Complexity: O(n^2) worst case | Space Complexity: O(min(n, 26))
#
# a = left pointer, b = right pointer, substring_list = current window as set.
# If s[b] not in window: expand right.
# If s[b] in window: shrink left until duplicate removed, then continue.
# max_length updated each iteration.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        a = 0
        b = 1
        substring_list = set(s[a])
        max_length = 1
        while b < len(s):
            if s[b] not in substring_list:
                substring_list.add(s[b])
                b += 1
            elif s[b] in substring_list:
                while s[a] != s[b]:
                    substring_list.remove(s[a])
                    a += 1
                substring_list.remove(s[a])
                a += 1
            if b - a > max_length:
                max_length = b - a
        return max_length
