# LeetCode 67 - Add Binary
# https://leetcode.com/problems/add-binary/
# Difficulty: Easy
# Approach: Built-in binary conversion one liner
# Time Complexity: O(n) | Space Complexity: O(n)


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a, 2) + int(b, 2))[2:]
