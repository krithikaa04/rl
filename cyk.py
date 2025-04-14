

from nltk import CFG, Nonterminal
from collections import defaultdict

# === Define grammar productions as strings ===
# Must be in CNF: A -> B C or A -> 'a'
grammar_rules = """
S -> NP VP
VP -> VP PP
VP -> V NP
VP -> V NP PP
PP -> P NP
NP -> 'John'
NP -> 'Mary'
NP -> 'Denver'
V -> 'called'
P -> 'from'
"""

# === Load grammar using CFG ===
def load_grammar(grammar_str):
    cfg = CFG.fromstring(grammar_str)
    return cfg.productions()

def build_reverse_map(productions):
    rhs_to_lhs = defaultdict(list)
    for prod in productions:
        rhs_to_lhs[prod.rhs()].append(prod.lhs())
    return rhs_to_lhs

# === CYK Algorithm ===
def cyk_parser(tokens, productions):
    n = len(tokens)
    T = [[set() for _ in range(n)] for _ in range(n)]
    B = [[dict() for _ in range(n)] for _ in range(n)]

    rhs_to_lhs = build_reverse_map(productions)

    # Fill diagonal (terminal rules)
    for j in range(n):
        word = tokens[j]
        for lhs in rhs_to_lhs.get((word,), []):
            T[j][j].add(lhs)
            B[j][j][lhs] = word

    # Fill upper triangle
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for B_sym in T[i][k]:
                    for C_sym in T[k + 1][j]:
                        for A_sym in rhs_to_lhs.get((B_sym, C_sym), []):
                            T[i][j].add(A_sym)
                            B[i][j][A_sym] = (k, B_sym, C_sym)

    return T, B

# === RUN ===
productions = load_grammar(grammar_rules)
tokens = "John called Mary from Denver".split()
T, B = cyk_parser(tokens, productions)

root = Nonterminal("S")
if root in T[0][-1]:
    print("Sentence is grammatically valid.")
else:
    print("Sentence is NOT valid in the language.")