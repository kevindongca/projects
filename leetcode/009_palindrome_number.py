class Solution:
    def isPalindrome(self, x: int) -> bool:
        temp = x
        number = 0
        while temp > 0:
            digit = temp % 10
            number = number * 10 + digit
            temp = temp // 10
        return number == x