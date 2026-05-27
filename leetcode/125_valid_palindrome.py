# LeetCode 125 - Valid Palindrome
# https://leetcode.com/problems/valid-palindrome/
# Difficulty: Easy
# Approach: Strip non-alphanumeric with regex, compare to reverse
# Time Complexity: O(n) | Space Complexity: O(n)

import re


class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r'[^a-zA-Z0-9]', '', s).lower().strip()
        return s[::-1] == s
