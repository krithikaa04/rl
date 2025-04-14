

#Probabilistic CYK
import nltk
from nltk import Tree, Nonterminal
from collections import defaultdict

# Sample PCFG training trees
trees = [
    "(S (NP John) (VP (VP (V called) (NP Mary)) (PP (P from) (NP Denver))))",
    "(S (NP John) (VP (V called) (NP Mary) (PP (P from) (NP Denver))))"
]

# Build rule probabilities
def extract_probabilistic_rules(tree_strings):
    rule_counts = defaultdict(int)
    lhs_counts = defaultdict(int)

    for s in tree_strings:
        for rule in Tree.fromstring(s).productions():
            rule_counts[rule] += 1
            lhs_counts[rule.lhs()] += 1

    return {rule: rule_counts[rule] / lhs_counts[rule.lhs()] for rule in rule_counts}

# Probabilistic CYK parser (structured like symbolic CYK)
def probabilistic_cyk(tokens, R):
    n = len(tokens)
    T = [[defaultdict(float) for _ in range(n)] for _ in range(n)]
    B = [[defaultdict(tuple) for _ in range(n)] for _ in range(n)]

    # Build reverse map: RHS -> list of (LHS, prob)
    rhs_to_lhs = defaultdict(list)
    for rule, prob in R.items():
        rhs_to_lhs[rule.rhs()].append((rule.lhs(), prob))

    # Fill diagonal
    for j in range(n):
        word = tokens[j]
        for lhs, prob in rhs_to_lhs.get((word,), []):
            T[j][j][lhs] = prob
            B[j][j][lhs] = word

    # CYK table filling
    for span in range(2, n+1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for B_sym, B_prob in T[i][k].items():
                    for C_sym, C_prob in T[k+1][j].items():
                        for A_sym, rule_prob in rhs_to_lhs.get((B_sym, C_sym), []):
                            total_prob = rule_prob * B_prob * C_prob
                            if total_prob > T[i][j][A_sym]:
                                T[i][j][A_sym] = total_prob
                                B[i][j][A_sym] = (k, B_sym, C_sym)

    return T, B

# Tree reconstruction
def build_tree(B, i, j, sym):
    val = B[i][j].get(sym)
    if val is None:
        return None
    if isinstance(val, str):
        return Tree(sym.symbol(), [val])
    k, B_sym, C_sym = val
    left = build_tree(B, i, k, B_sym)
    right = build_tree(B, k+1, j, C_sym)
    return Tree(sym.symbol(), [left, right])

# === RUN ===
R = extract_probabilistic_rules(trees)
tokens = "John called Mary from Denver".split()
T, B = probabilistic_cyk(tokens, R)

root = Nonterminal("S")
if root in T[0][-1]:
    print(f"\nSentence is grammatically valid. Probability: {T[0][-1][root]:.6f}")
    tree = build_tree(B, 0, len(tokens)-1, root)
    print("\nMost Probable Parse Tree:")
    tree.pretty_print()
else:
    print("Sentence is not in the language.")