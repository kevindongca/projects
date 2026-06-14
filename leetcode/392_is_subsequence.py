# LeetCode 392 - Is Subsequence
# https://leetcode.com/problems/is-subsequence/
# Difficulty: Easy
# Approach: Greedy scan — match s characters in order while iterating t
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Track index into s. For each letter in t, if it matches s[s_indice], advance.
# If s_indice reaches len(s), all characters matched in order -> True.
# Empty string is always a subsequence (base case).

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if s == "":
            return True
        s_indice = 0
        for letter in t:
            if letter == s[s_indice]:
                s_indice += 1
            if s_indice == len(s):
                return True
        return False
