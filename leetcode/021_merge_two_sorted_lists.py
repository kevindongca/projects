# Dummy node approach — build merged list by comparing heads each iteration
# Time: O(n+m) | Space: O(1)
class Solution:
    def mergeTwoLists(self, list1, list2):
        dummy = ListNode(0)
        current = dummy

        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # attach remaining nodes from whichever list still has elements
        if list1:
            current.next = list1
        if list2:
            current.next = list2

        return dummy.next
