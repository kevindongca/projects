# LeetCode 203 - Remove Linked List Elements
# https://leetcode.com/problems/remove-linked-list-elements/
# Difficulty: Easy
# Approach: Dummy node + prev/curr pointers
# Time Complexity: O(n) | Space Complexity: O(1)
#
# Dummy node before head lets us handle head deletion without a special case.
# prev trails curr by one node. When curr.val == val, skip it: prev.next = curr.next.
# When curr.val != val, advance prev. Always advance curr.
# Return dummy.next as the new head.

from typing import Optional


class Solution:
    def removeElements(self, head: Optional['ListNode'], val: int) -> Optional['ListNode']:
        dummy = ListNode(0, head)
        prev, curr = dummy, head
        while curr is not None:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummy.next
