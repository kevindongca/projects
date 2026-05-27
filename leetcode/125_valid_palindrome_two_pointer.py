# LeetCode 125 - Valid Palindrome
# https://leetcode.com/problems/valid-palindrome/
# Difficulty: Easy
# Approach 1: regex clean + reverse compare  O(n) time O(n) space
# Approach 2: two pointer (below)            O(n) time O(n) space

import re


class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r'[^a-zA-Z0-9]', '', s).lower().strip()
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True
