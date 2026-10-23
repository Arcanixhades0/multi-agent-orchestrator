"""
Evaluation tasks. Each needs research (what is X, edge cases) AND code.

`hidden_tests` are never shown to the agents. A task counts as completed only if
the final code passes them, so "completion rate" is an objective number,
not an LLM's opinion.
"""

TASKS = [
    {
        "id": "levenshtein",
        "task": "Explain edit distance and implement levenshtein(a: str, b: str) -> int.",
        "hidden_tests": "assert levenshtein('kitten','sitting')==3\nassert levenshtein('','abc')==3\nassert levenshtein('same','same')==0",
    },
    {
        "id": "rle",
        "task": "Explain run-length encoding and implement rle_encode(s: str) -> str returning e.g. 'a3b2' for 'aaabb', and rle_decode(s: str) -> str.",
        "hidden_tests": "assert rle_encode('aaabb')=='a3b2'\nassert rle_decode('a3b2')=='aaabb'\nassert rle_encode('')==''\nassert rle_decode(rle_encode('abcccc'))=='abcccc'",
    },
    {
        "id": "roman",
        "task": "Research how Roman numerals work including subtractive notation, then implement to_roman(n: int) -> str for 1..3999 and from_roman(s: str) -> int.",
        "hidden_tests": "assert to_roman(1994)=='MCMXCIV'\nassert from_roman('MCMXCIV')==1994\nassert to_roman(4)=='IV'\nassert all(from_roman(to_roman(i))==i for i in range(1,4000))",
    },
    {
        "id": "luhn",
        "task": "Explain the Luhn checksum algorithm and implement luhn_valid(number: str) -> bool that ignores spaces.",
        "hidden_tests": "assert luhn_valid('4539 3195 0343 6467')\nassert not luhn_valid('8273 1232 7352 0569')\nassert luhn_valid('79927398713')",
    },
    {
        "id": "lru",
        "task": "Explain how an LRU cache works and implement class LRUCache(capacity) with get(key) returning -1 if missing, and put(key, value).",
        "hidden_tests": "c=LRUCache(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1\nassert c.get(3)==3 and c.get(4)==4",
    },
    {
        "id": "dijkstra",
        "task": "Explain Dijkstra's algorithm and implement dijkstra(graph: dict, start) -> dict of shortest distances, where graph maps node -> list of (neighbor, weight). Unreachable nodes are omitted.",
        "hidden_tests": "g={'a':[('b',1),('c',4)],'b':[('c',2),('d',5)],'c':[('d',1)],'d':[],'z':[]}\nd=dijkstra(g,'a')\nassert d['d']==4 and d['c']==3 and d['a']==0\nassert 'z' not in d",
    },
    {
        "id": "balanced",
        "task": "Research how to check bracket balancing with a stack and implement is_balanced(s: str) -> bool for (), [], {} ignoring other characters.",
        "hidden_tests": "assert is_balanced('([]{})')\nassert not is_balanced('([)]')\nassert is_balanced('')\nassert not is_balanced('((')\nassert is_balanced('a(b)c')",
    },
    {
        "id": "binary_search",
        "task": "Explain binary search and its off-by-one pitfalls, then implement lower_bound(arr: list, x) -> int returning the first index where arr[i] >= x (len(arr) if none).",
        "hidden_tests": "assert lower_bound([1,2,2,3],2)==1\nassert lower_bound([1,2,3],4)==3\nassert lower_bound([],1)==0\nassert lower_bound([5],1)==0",
    },
    {
        "id": "sieve",
        "task": "Explain the Sieve of Eratosthenes and implement primes_up_to(n: int) -> list of primes <= n.",
        "hidden_tests": "assert primes_up_to(30)==[2,3,5,7,11,13,17,19,23,29]\nassert primes_up_to(1)==[]\nassert primes_up_to(2)==[2]",
    },
    {
        "id": "topo",
        "task": "Explain topological sorting and implement topo_sort(deps: dict) -> list where deps maps node -> list of nodes it depends on (dependencies come first). Raise ValueError on a cycle.",
        "hidden_tests": "o=topo_sort({'app':['lib','util'],'lib':['util'],'util':[]})\nassert o.index('util')<o.index('lib')<o.index('app')\ntry:\n    topo_sort({'a':['b'],'b':['a']}); assert False\nexcept ValueError: pass",
    },
    {
        "id": "word_freq",
        "task": "Research text normalisation (case, punctuation) and implement top_words(text: str, k: int) -> list of (word, count) for the k most common words, lowercase, punctuation removed, ties broken alphabetically.",
        "hidden_tests": "assert top_words('The cat. the dog! A cat?',2)==[('cat',2),('the',2)]\nassert top_words('',3)==[]",
    },
    {
        "id": "matrix_spiral",
        "task": "Explain how to traverse a matrix in spiral order and implement spiral(matrix: list[list]) -> list.",
        "hidden_tests": "assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]\nassert spiral([])==[]\nassert spiral([[1,2],[3,4],[5,6]])==[1,2,4,6,5,3]",
    },
    {
        "id": "interval_merge",
        "task": "Research interval merging and implement merge_intervals(intervals: list[tuple]) -> list[tuple] merging overlapping or touching intervals, sorted.",
        "hidden_tests": "assert merge_intervals([(1,3),(2,6),(8,10)])==[(1,6),(8,10)]\nassert merge_intervals([(1,4),(4,5)])==[(1,5)]\nassert merge_intervals([])==[]",
    },
    {
        "id": "base64",
        "task": "Explain how Base64 encoding works and implement b64encode(data: bytes) -> str WITHOUT using the base64 module.",
        "hidden_tests": "import base64\nfor s in [b'', b'f', b'fo', b'foo', b'hello world!']:\n    assert b64encode(s)==base64.b64encode(s).decode()",
    },
    {
        "id": "kadane",
        "task": "Explain Kadane's algorithm and implement max_subarray(nums: list[int]) -> int (largest sum of a non-empty contiguous subarray).",
        "hidden_tests": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6\nassert max_subarray([-3,-1,-2])==-1\nassert max_subarray([5])==5",
    },
    {
        "id": "anagram_groups",
        "task": "Research efficient anagram detection and implement group_anagrams(words: list[str]) -> list[list[str]], each group sorted, groups sorted by first word.",
        "hidden_tests": "assert group_anagrams(['eat','tea','tan','ate','nat','bat'])==[['ate','eat','tea'],['bat'],['nat','tan']]\nassert group_anagrams([])==[]",
    },
    {
        "id": "cosine",
        "task": "Explain cosine similarity as used for embeddings and implement cosine_similarity(a: list[float], b: list[float]) -> float, returning 0.0 if either vector is all zeros.",
        "hidden_tests": "assert abs(cosine_similarity([1,0],[0,1]))<1e-9\nassert abs(cosine_similarity([1,2],[2,4])-1)<1e-9\nassert cosine_similarity([0,0],[1,1])==0.0",
    },
    {
        "id": "rrf",
        "task": "Research Reciprocal Rank Fusion for search and implement rrf(rankings: list[list[str]], k: int = 60) -> list[str] returning ids sorted by fused score (ties by id).",
        "hidden_tests": "r=rrf([['a','b','c'],['c','d']])\nassert r[0]=='c'\nassert set(r)=={'a','b','c','d'}\nassert rrf([])==[]",
    },
    {
        "id": "bm25_idf",
        "task": "Explain BM25's IDF term and implement bm25_idf(docs: list[list[str]], term: str) -> float using log((N - n + 0.5)/(n + 0.5) + 1).",
        "hidden_tests": "import math\nd=[['a','b'],['a'],['c']]\nassert abs(bm25_idf(d,'a')-math.log((3-2+0.5)/(2+0.5)+1))<1e-9\nassert abs(bm25_idf(d,'z')-math.log((3+0.5)/0.5+1))<1e-9",
    },
    {
        "id": "token_bucket",
        "task": "Explain the token bucket rate-limiting algorithm and implement class TokenBucket(capacity, refill_per_sec) with allow(now: float) -> bool, starting full, using the passed-in time instead of the clock.",
        "hidden_tests": "b=TokenBucket(2,1)\nassert b.allow(0) and b.allow(0)\nassert not b.allow(0)\nassert b.allow(1.0)\nassert not b.allow(1.0)\nassert b.allow(10) and b.allow(10) and not b.allow(10)",
    },
]
