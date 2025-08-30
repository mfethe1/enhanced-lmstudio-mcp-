import re, json
code=open('paste.txt','r',encoding='utf-8').read()
todos=re.findall(r'#\s*TODO.*', code, flags=re.IGNORECASE)
len_todos=len(todos)
{'todo_count':len_todos,'sample':todos[:5]}