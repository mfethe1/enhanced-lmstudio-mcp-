import statistics, ast, json, textwrap
code = open('paste.txt','r',encoding='utf-8').read()
func_lens=[]
func_names=[]
cur_name=None
for node in ast.walk(ast.parse(code)):
    if isinstance(node, ast.FunctionDef):
        start=node.lineno
        end=getattr(node,'end_lineno',start)
        func_lens.append(end-start+1)
        func_names.append((node.name,end-start+1))

# top 10 longest
longest=sorted(func_names,key=lambda x:-x[1])[:10]
top_longest=longest
{
 'average':statistics.mean(func_lens),
 'median':statistics.median(func_lens),
 'max':max(func_lens),
 'longest':top_longest
}