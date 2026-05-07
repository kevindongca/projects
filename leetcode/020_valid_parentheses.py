# Stack approach — push opening brackets, match closing brackets against top of stack
# Time: O(n) | Space: O(n)
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        matching = {')': '(', ']': '[', '}': '{'}
        for entry in s:
            if entry in "({[":
                stack.append(entry)
            elif not stack or stack[-1] != matching[entry]:
                return False
            else:
                stack.pop()
        return len(stack) == 0
