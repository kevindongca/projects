class Solution:
    def romanToInt(self, s: str) -> int:
        values = {
            "I": 1, "V": 5, "X": 10, "L": 50,
            "C": 100, "D": 500, "M": 1000
        }
        n = 0
        for x in range(len(s)):
            if x + 1 < len(s) and values[s[x]] < values[s[x+1]]:
                n -= values[s[x]]
            else:
                n += values[s[x]]
        return n