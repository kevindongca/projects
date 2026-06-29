# LeetCode 876 - Middle of the Linked List
# https://leetcode.com/problems/middle-of-the-linked-list/
# Difficulty: Easy
# Approach: Fast/slow pointers — fast moves 2 steps, slow moves 1
# Time Complexity: O(n) | Space Complexity: O(1)
#
# When fast reaches the end, slow is at the middle.
# For even-length lists, returns the second middle node (e.g. [1,2,3,4] -> node 3).
# Same tortoise and hare mechanic as 141/142 but no cycle — fast hits None naturally.

from typing import Optional


class Solution:
    def middleNode(self, head: Optional['ListNode']) -> Optional['ListNode']:
        fast, slow = head, head
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow
