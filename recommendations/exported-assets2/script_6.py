from pathlib import Path
import importlib.util, ast, inspect, textwrap, os, re, json, sys
code = open('paste.txt','r', encoding='utf-8').read()

def extract_function_source(name):
    pattern = rf'def {name}\s*\(.*\):'
    m = re.search(pattern, code)
    if not m:
        return None
    start = m.start()
    indent = len(re.match(r'\s*', code[start:]).group())
    lines = code[start:].splitlines()
    collected=[]
    for ln in lines:
        collected.append(ln)
        if ln.strip()=='' and len(collected)>1 and not any(line.strip() for line in lines[len(collected):]):
            break
    return '\n'.join(collected[:40])

_safe_path_src = extract_function_source('_safe_path')
print(_safe_path_src)