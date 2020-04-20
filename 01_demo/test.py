class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteDuplicates(head):
        if not head:return 0
        a=head
        while a and a.next:
            if a.val==a.next.val:
                a.next=a.next.next
            else:
                a=a.next
        return head
