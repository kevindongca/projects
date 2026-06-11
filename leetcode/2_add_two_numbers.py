# LeetCode 2 - Add Two Numbers
# https://leetcode.com/problems/add-two-numbers/
# Difficulty: Medium
# Approach: Simulate addition digit by digit with carry, dummy head node
# Time Complexity: O(max(m,n)) | Space Complexity: O(max(m,n))
#
# Lists are stored in reverse order so we add from head (least significant digit).
# Each iteration: start total from carry, add l1.val and l2.val if they exist.
# num = total % 10 is the current digit, carry = total // 10 for next iteration.
# Loop continues while either list has nodes remaining OR carry is non-zero.
# Dummy node avoids special-casing the head; return dummy.next.

from typing import Optional


class Solution:
    def addTwoNumbers(self, l1: Optional['ListNode'], l2: Optional['ListNode']) -> Optional['ListNode']:
        dummy = ListNode()
        result = dummy
        total = carry = 0
        while l1 or l2 or carry:
            total = carry
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next
            num = total % 10
            carry = total // 10
            dummy.next = ListNode(num)
            dummy = dummy.next
        return result.next
