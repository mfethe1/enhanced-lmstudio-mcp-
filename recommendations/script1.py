import ast, textwrap, json, re, os, hashlib, math, sys, inspect, tokenize, io, pathlib, collections, statistics
code = open('paste.txt', 'r', encoding='utf-8').read()
lines = code.splitlines()
num_lines = len(lines)
num_blank = sum(1 for l in lines if l.strip()=='' )
num_comments = sum(1 for l in lines if l.strip().startswith('#'))
num_funcs = 0
num_classes = 0
max_func_len = 0
func_lens = []

try:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            num_funcs += 1
            start = node.lineno
            end = max(getattr(node, 'end_lineno', start), start)
            func_lens.append(end-start+1)
            max_func_len = max(max_func_len, end-start+1)
        elif isinstance(node, ast.ClassDef):
            num_classes += 1
except Exception as e:
    pass
avg_func_len = statistics.mean(func_lens) if func_lens else 0
stats = {
    'lines': num_lines,
    'blank_lines': num_blank,
    'comment_lines': num_comments,
    'functions': num_funcs,
    'classes': num_classes,
    'max_func_len': max_func_len,
    'avg_func_len': avg_func_len,
}
stats