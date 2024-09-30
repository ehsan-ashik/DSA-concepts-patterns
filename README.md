# Important DSA concepts/patterns

Important data structures and algorithmic concepts that can help solve problems easily. 


## Monotonic Queue/Stack 

Keeps values in the queue/stack either monotonically increasing or decreasing order. Can help when we need to keep track of a min/max value in a range for multiple iterations.

**Example Problems**
1. [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/)
2. [Min Stack](https://leetcode.com/problems/min-stack/description/)
3. [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/)
4. [Car Fleet](https://leetcode.com/problems/car-fleet/description/)

## Binary Search Left-most/Right-most

Binary search can be formated based on the specific need of the problem. For example, if it is expected to find the left-most or right-most occurrance of some target in a sorted array, binary search can be formatted accordingly. 

In addition, this formats can be helpful to ensure no *infinite loop* issue in the code.

#### 1. Finding the left-most occurance of an element:

```python
	left, right = 0, len(nums) - 1
	
	while left < right:
		mid = left + (right - left) // 2 # this format ensures no overflow compared to (left + right) // 2
		if nums[mid] >= target: # if an occurance is found, we are shifting right to the mid, expecting other occurances in the left
			right = mid 
		else:
			left = mid + 1
```

#### 2.  the right-most occurance of an element:

```python
	left, right = 0, len(nums) - 1
	
	while left < right:
		mid = left + (right - left) // 2 + 1 # notice the + 1 to the mid. This will ensure that mid is always biased towards right and prevent infinite loop
		if nums[mid] > target:
			right = mid - 1
		else: # if an occurance is found, we are shifting left to the mid, expecting other occurances in the right
			left = mid
```

**Example Problems**
1. [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
2. [Time Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/description/)



## Linked List Reversal

Linked list reverse can be done both **iteratively** and **recursively**. Iterative reversal needs only a `prev` pointer (indicates previous node) that initially points to `None` and gets updated accordingly. At the end of the iteration, the `prev` points to the reversed list. Here is a sample code that can be used for many deferent problems.

```python
	prev = None
	while head:
		nxt = head.next
		head.next = prev
		prev = head
		head = nxt	
	return prev
```

**Example Problems**
1. [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)



## Floyd Cycle Detection

This is a helpful way to both **detect** and **start** of a cycle in linked list. It uses a *two-pointer apporach*, known as **slow** and **fast** pointers. 

#### Approach for detecting cycles: 

Start slow and fast at the same beginning node. *Setting it to the same head node is crucial for detecting the point where cycle is created*. In each iteration, increament slow by one and fast by two. If slow and fast meets, there is a cycle in the linked list. 


#### Approach for finding out the node where the cycle begins:

After detecting circle, iterate through **head** and **slow** by one, until they meets. The point where they meet is the start of the cycle. 

**Example Problems**
1. [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/)
2. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
3. [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/description/)



## Trie and TrieNode

Trie, also known as *Prefix tree* is a data structure to store words. This data structure can be helpful to search and find matches in a word cloud efficiently. 

Typically a `TrieNode` contains two properties: `children` and `isWord`. A Trie is a collection of TrieNodes starting from the root node. Additionally, a third property `refs` can be helpful in scenarios where all possible matching words may need to be retried. When a word is added, references of all the nodes for the word in incremented. On the other hand, when a word is removed its reference is decremented. In a check we can skip nodes whose refs are 0, indicating that all the words starting from the node is traversed. 

```python
	class TrieNode:
		def __init__(self):
			self.children = dict() # empty dict when initializing a trie node
			self.isWord = False # if this node indicates the end of an word, set it to true
			self.refs = 0 # increament refs when a word is added and decrement when a word is removed
	
	class Trie:
		def __init__(self):
			self.root = TrieNode() # the root node of the Trie
```

**Example Problems**
1. [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/)
2. [Word Search](https://leetcode.com/problems/word-search/description/)
3. [Word Search II](https://leetcode.com/problems/word-search-ii/description/)



## UnionFind

Also known as **Disjoint Set**, this algorithm can help efficiently merge disjoint items into one by keeping a list of nodes' `parents` and their current `ranks`.

The `find` method iterate through the parents and find the parent of a given node. With the `path compression` cache implemented, finding parent of a node takes on average `O(log n)` time.

The `Union` method merge two disjoint nodes into one, by merging their parents. An efficient merging takes into account the `rank` of the two nodes and merge the node with lower rank with the node that has higher rank.

Here is an example implementation of this data structure. 

```python
	class UnionFind:
		def __init__(self, n): # n is the size of the disjoint sets/nodes
			self.parent = [i  for i in range(n)] # initiallly each node is it's parent
			self.rank = [1] * n # initially each node is of rank/size of length 1
		
		def find(self, node):
			if node != self.parent[node]: # iterate till the true parent is found
				self.parent[node] = self.find(self.parent[node]) # recursively find the parent and cache (update old parents) for efficiency
			return self.parent[node]
	
		class union(self, node1, node2): # merge two nodes, meaning make their parents same and update their ranks
			parent1 = self.find(node1)
			parent2 = self.find(node2)
			
			if parent1 == parent2: # if both nodes have the same parent no union is needed, return False
				return False
			
			if self.rank[parent1] >= self.rank[parent2]: # efficiently merge based on their rank
				self.rank[parent1] += self.rank[parent2] # parent1 is of higher rank. Add parent2's rank to make the rank even higher
				self.parent[parent2] = parent1 # update node2's parent's reference
			else:
				self.rank[parent2] += self.rank[parent1] # parent2 is of higher rank. Do the opposite
				self.parent[parent1] = parent2 # update parent1's parent to parent2, otherwise, merging the two nodes
			
			return True
```

**Example Problems**
1. [Redundant Connection](https://leetcode.com/problems/redundant-connection/description/)
2. [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/description/)



## Minimum Spanning Tree (MST)

Minimum Spanning Tree (MST) is a subset of edges in a graph that connects all the vertices of the grapgh with minimum possible total weight. To find out the MST of a Graph, we can use **Kruskal's Algorithm**. Kruskal algorithm greedily picks the minimum weight edge without creating a cycle in the graph. Hence a **MinHeap** can help to select the next minimum edge. To detect whether a picked edge has created a cycle in the graph, we can leverage the **UnionFind** data structure. 

**Example Problems**
1. [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/description/)



## Shortest path in a Directed/Undirected Graph:

**Dijakstra's Algorithm** can help find the shortest path (from a source to a destination node) in a directed or undirected weighted graph that does not contain *negative edge weights*. The algorithm starts from the *source* node and greedily pick the next node with the shortest distance. If a shorter path to a node is found, the shortest distance is updated accordingly. We can leverage a **Min Heap** to pick the next node until the *destination* node is obtained. If a new shorter path for a node is found, we update it's distance and put it in the min heap for subsequent runs. 

**Example Problems**
1. [Network Delay Time](https://leetcode.com/problems/network-delay-time/description/)



## Topological Sorting

Topological sorting/ordering is a linear ordering of the *Nodes* in an **Directed Acyclic Graph (DAG)** that satisfies all the edge direction constrainsts. We can use *Bredth First Search (BFS)* with *post-order traversal* to get the linear ordering of the nodes in the DAG. Once all the children of a node are visited, the node can be added to the *stack* till there are unvisited nodes in the graph. After that, popping the nodes will give the *topological order* of the nodes.

The algorithm works even when there are multiple DAGs in the graph as disconnected components. 

Topological ordering can be helpful to to detect cycles in a graph.

**Example Problems**
1. [Alien Dictionary](https://neetcode.io/problems/foreign-dictionary)
2. [Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/description/)
