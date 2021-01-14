import graphviz

def gv(s: str):
    return graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')