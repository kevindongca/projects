# LeetCode 67 - Add Binary
# https://leetcode.com/problems/add-binary/
# Difficulty: Easy
# Approach: Manual binary to decimal conversion, add, convert back
# Time Complexity: O(n) | Space Complexity: O(n)


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        c, d = 0, 0
        for i in range(len(a)-1, -1, -1):
            c += int(a[i]) * 2**(len(a) - 1 - i)
        for i in range(len(b)-1, -1, -1):
            d += int(b[i]) * 2**(len(b) - 1 - i)
        e = c + d
        f = ""
        while e > 0:
            f += str(e % 2)
            e = e // 2
        if f == "":
            return "0"
        return f[::-1]
