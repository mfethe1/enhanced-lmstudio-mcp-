import ast, collections, token, tokenize, io, json
code=open('paste.txt','r',encoding='utf-8').read()
module_ast=ast.parse(code)
imports=[]
used=set()
class ImportVisitor(ast.NodeVisitor):
    def visit_Import(self,node):
        for alias in node.names:
            imports.append(alias.name.split('.')[0])
    def visit_ImportFrom(self,node):
        if node.module:
            imports.append(node.module.split('.')[0])
    def visit_Name(self,node):
        used.add(node.id)
ImportVisitor().visit(module_ast)
m_counts=collections.Counter(imports)
unused=[m for m in m_counts if m not in used]
{
 'total_imports':len(imports),
 'unique_imports':len(set(imports)),
 'unused_imports':unused[:20]
}