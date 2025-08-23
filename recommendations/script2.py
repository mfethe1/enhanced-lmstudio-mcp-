import ast, statistics, collections, tokenize, io, textwrap, json, re, math
code = open('paste.txt', 'r', encoding='utf-8').read()

complexities = {}

def count_branches(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler, ast.BoolOp)):  # BoolOp separate maybe overshoot
            count += 1
    return count + 1  # base complexity 1

try:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            complexities[node.name] = count_branches(node)
except Exception as e:
    complexities = {}

# Summaries
all_vals = list(complexities.values())
summary = {
    'avg_cyclomatic': statistics.mean(all_vals) if all_vals else 0,
    'max_cyclomatic': max(all_vals) if all_vals else 0,
    'functions_over_15': sum(1 for v in all_vals if v>15),
    'functions_over_10': sum(1 for v in all_vals if v>10),
}
summary