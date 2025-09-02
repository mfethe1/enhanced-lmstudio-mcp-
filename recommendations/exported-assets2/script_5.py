import re, json
code = open('paste.txt','r', encoding='utf-8').read()
logging_calls = re.findall(r'logger\.(debug|info|warning|error|critical)\(', code)
len_logging=len(logging_calls)

{'logging_calls': len_logging, 'levels_present': sorted(set(logging_calls))}