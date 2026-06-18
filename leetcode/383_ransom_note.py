# LeetCode 383 - Ransom Note
# https://leetcode.com/problems/ransom-note/
# Difficulty: Easy
# Approach: Counter subtraction
# Time Complexity: O(n) | Space Complexity: O(1) (at most 26 keys)
#
# Counter(ransomNote) - Counter(magazine) subtracts character counts.
# Counter subtraction drops zero/negative results (keeps only unmet demand).
# If the result is empty, magazine covers every character ransomNote needs -> True.
# not {} == True, not {remaining chars} == False.

from collections import Counter


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        return not (Counter(ransomNote) - Counter(magazine))
