import re, json, os, textwrap
code = open('paste.txt','r', encoding='utf-8').read()
subproc_calls = re.findall(r'subprocess\.(run|call|Popen|check_call|check_output)\s*\(([^\)]*)\)', code)
uses_shell_true = []
for func,arg in subproc_calls:
    if 'shell=True' in arg or 'shell = True' in arg:
        uses_shell_true.append((func, arg.strip()[:120]))
len_calls=len(subproc_calls)
len_shell_true=len(uses_shell_true)
{'total_subprocess_calls': len_calls, 'shell_true_calls': len_shell_true, 'examples': uses_shell_true[:3]}