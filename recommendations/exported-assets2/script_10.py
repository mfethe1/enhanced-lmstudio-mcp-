import ast, json, statistics
code=open('paste.txt','r',encoding='utf-8').read()

def complexity(node):
    count=1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.Try, ast.With, ast.ExceptHandler, ast.BoolOp)):
            count+=1
    return count

complexities=[]
for node in ast.walk(ast.parse(code)):
    if isinstance(node, ast.FunctionDef):
        c=complexity(node)
        complexities.append({'function': node.name,'cc':c})

# bucket counts
import collections
bins={ '1-5':0,'6-10':0,'11-15':0,'16+':0}
for item in complexities:
    if item['cc']<=5:
        bins['1-5']+=1
    elif item['cc']<=10:
        bins['6-10']+=1
    elif item['cc']<=15:
        bins['11-15']+=1
    else:
        bins['16+']+=1
print(json.dumps(bins))