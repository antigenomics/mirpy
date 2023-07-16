from collections import defaultdict
from collections.abc import Set

class Trie(Set):
    """Taken from https://codereview.stackexchange.com/a/142903
    A set of strings implemented using a trie."""

    #https://habr.com/ru/articles/681940/

    def __init__(self, iterable=()):
        self._children = defaultdict(Trie)
        self._end = False
        self._len = 0
        for element in iterable:
            self.add(element)

    def add(self, element):
        node = self
        for s in element:
            node = node._children[s]
        if not node._end:
            node._end = True 
            node._len += 1
            node = self
            for s in element:
                node._len += 1
                node = node._children[s]

    def __contains__(self, element):
        node = self
        for k in element:
            if k not in node._children:
                return False
            node = node._children[k]
        return node._end

    def search(self, term):
        """Generate the elements of the set matching the search term, which
        may include wildcards. Wildcards are: '.' for any symbol,
        '-' for any string of symbols (arbitrary length)
        """
        def _add(indexes, i):
            # Add i to set of indexes into term, taking account of wildcards.
            while i < len(term) and term[i] == '*':
                indexes.add(i)
                i += 1
            indexes.add(i)

        indexes = set()
        _add(indexes, 0)
        if self._end and len(term) in indexes:
            yield ''

        indexes_stack = [indexes] # Stack of sets of indexes into term.
        element = ['']            # Current element reached in search.
        iter_stack = [iter(self._children.items())] # Stack of iterators.

        while iter_stack:
            for k, node in iter_stack[-1]:
                new_indexes = set()
                for i in indexes_stack[-1]:
                    if i >= len(term):
                        continue
                    elif term[i] == '.':
                        _add(new_indexes, i)
                    elif term[i] == '-' or term[i] == k:
                        _add(new_indexes, i + 1)
                if new_indexes:
                    element.append(k)
                    if node._end and len(term) in new_indexes:
                        yield ''.join(element)
                    indexes_stack.append(new_indexes)
                    iter_stack.append(iter(node._children.items()))
                    break
            else:
                element.pop()
                indexes_stack.pop()
                iter_stack.pop()

    def __iter__(self):
        element = ['']
        stack = [iter([('', self)])]
        while stack:
            for k, node in stack[-1]:
                element.append(k)
                if node._end:
                    yield ''.join(element)
                stack.append(iter(node._children.items()))
                break
            else:
                element.pop()
                stack.pop()

    def __len__(self):
        return self._len