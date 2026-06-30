# LeetCode 49 - Group Anagrams
# https://leetcode.com/problems/group-anagrams/
# Difficulty: Medium
# Approach: Hashmap keyed by sorted word (anagrams share the same sorted form)
# Time Complexity: O(n * k log k) where k is max word length | Space Complexity: O(n*k)
#
# Sorting each word gives a canonical key — all anagrams of each other sort
# to the identical string (e.g. "eat", "tea", "ate" all sort to "aet").
# Group words under that key, return all groups as a list.

from typing import List


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = {}
        for word in strs:
            key = ''.join(sorted(word))
            if key not in groups:
                groups[key] = []
            groups[key].append(word)
        return list(groups.values())
