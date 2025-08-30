import os, json
os.environ['ENHANCED_STORAGE'] = '1'

from enhanced_mcp_storage_v2 import StorageSelector
from enhanced_mcp_storage import EnhancedMCPStorage

legacy = EnhancedMCPStorage()
st = StorageSelector(legacy)

# Clean any old test keys
for k in ['k1', 'semantic_summary_test']:
    try:
        st.delete_memory(k)
    except Exception:
        pass

# Store general and semantic entries
assert st.store_memory('k1', 'v1', 'general')
assert st.store_memory('semantic_summary_test', json.dumps({'summary': 'Deep analysis of router heuristics for code review'}), 'semantic_memory')

rows_general = st.retrieve_memory(category='general', limit=10)
rows_sem = st.retrieve_memory(category='semantic_memory', limit=10)
print('GENERAL_ROWS', len(rows_general))
print('SEMANTIC_ROWS', len(rows_sem))

rows_search = st.retrieve_memory(category='general', search_term='v1', limit=10)
print('SEARCH_ROWS', len(rows_search))

# Now test the server tool handlers
import server as srv
server = srv.EnhancedLMStudioMCPServer()

# Direct retrieve_memory tool
out = srv.handle_memory_retrieve({'category': 'general'}, server)
print('TOOL_RETRIEVE_OUTPUT_PREFIX', out[:60].replace('\n',' '))

# Semantic retrieval tool
out_sem = srv.handle_memory_retrieve_semantic({'query': 'router heuristics'}, server)
print('TOOL_SEMANTIC_RESULTS_COUNT', len(out_sem.get('results', [])))
print('TOOL_SEMANTIC_TOP', out_sem.get('results', [])[:1])

print('OK')

