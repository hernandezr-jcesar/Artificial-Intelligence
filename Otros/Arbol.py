import pydot as dot

tuplas = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

arbol = dot.Graph()

for tupla in tuplas:
    arbol.add_node(tupla[0])

for tupla in tuplas:
    for hijo in tupla[1:]:
        arbol.add_edge(tupla[0], hijo)

dot.render("arbol.pdf")