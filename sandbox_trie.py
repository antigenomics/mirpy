from mir.distances import search

t = search.Trie()
t.add('attagaca')
t.add('attacaca')
t.add('attacccca')

print([x for x in t.search('attagaca')])
print([x for x in t.search('atta?aca')])
print([x for x in t.search('atta*a')])

