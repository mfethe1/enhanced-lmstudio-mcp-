import re, json, pathlib, ast
code = open('paste.txt','r',encoding='utf-8').read()
import_usage = set(re.findall(r'import\s+([\w_]+)', code))
pathlib_used = 'pathlib' in import_usage or 'Path(' in code
safe_path_function = '_safe_path' in code
safe_directory = '_safe_directory' in code
{'pathlib_used': pathlib_used, 'safe_path_func_defined': safe_path_function, 'safe_directory_func_defined': safe_directory}