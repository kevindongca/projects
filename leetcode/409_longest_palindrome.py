# LeetCode 409 - Longest Palindrome
# https://leetcode.com/problems/longest-palindrome/
# Difficulty: Easy
# Approach: Count character frequencies, pair up even counts, allow one odd center
# Time Complexity: O(n) | Space Complexity: O(1) (at most 52 unique chars)
#
# For each unique character, take as many pairs as possible: (count // 2) * 2.
# If any character has an odd count, one can sit in the center -> add 1.
# extra stays 1 once set; multiple odd-count chars all donate pairs but only
# one gets the center slot.

class Solution:
    def longestPalindrome(self, s: str) -> int:
        pal_length = 0
        extra = 0
        unique = set(s)
        for item in unique:
            char_count = s.count(item)
            if char_count % 2 == 1:
                extra = 1
            pal_length += (char_count // 2) * 2
        return pal_length + extra
