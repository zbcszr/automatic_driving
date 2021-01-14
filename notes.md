## how to generate data??
dataset h5 document

1. 使用递归
要逆序打印链表 1->2->3（3,2,1)，可以先逆序打印链表 2->3(3,2)，最后再打印第一个节点 1。而链表 2->3 可以看成一个新的链表，要逆序打印该链表可以继续使用求解函数，也就是在求解函数中调用自己，这就是递归函数。

public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
    ArrayList<Integer> ret = new ArrayList<>();
    if (listNode != null) {
        ret.addAll(printListFromTailToHead(listNode.next));
        ret.add(listNode.val);
    }
    return ret;
}
2. 使用头插法
头插法顾名思义是将节点插入到头部：在遍历原始链表时，将当前节点插入新链表的头部，使其成为第一个节点。

链表的操作需要维护后继关系，例如在某个节点 node1 之后插入一个节点 node2，我们可以通过修改后继关系来实现：

node3 = node1.next;
node2.next = node3;
node1.next = node2;
