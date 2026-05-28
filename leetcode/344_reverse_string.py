# LeetCode 344 - Reverse String
# https://leetcode.com/problems/reverse-string/
# Difficulty: Easy
# Approach: In-place list reverse using built-in
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Alternative two pointer approach:
# l, r = 0, len(s) - 1
# while l < r:
#     s[l], s[r] = s[r], s[l]
#     l += 1
#     r -= 1

from typing import List


class Solution:
    def reverseString(self, s: List[str]) -> None:
        s.reverse()
