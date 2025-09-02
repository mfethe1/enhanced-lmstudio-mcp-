import ast, json, statistics
code=open('paste.txt','r',encoding='utf-8').read()
func_data=[]
for node in ast.walk(ast.parse(code)):
    if isinstance(node, ast.FunctionDef):
        start=node.lineno
        end=getattr(node,'end_lineno',start)
        func_data.append({'function': node.name, 'length': end-start+1})
# top 15 length
top=sorted(func_data,key=lambda x:-x['length'])[:15]
print(json.dumps(top))