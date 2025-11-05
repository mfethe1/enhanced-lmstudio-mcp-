#!/usr/bin/env python3
"""Debug script to identify which imports are causing hangs."""
import sys
import time

def test_import(module_name, description=""):
    """Test importing a module and report timing."""
    print(f"Testing import: {module_name} {description}")
    start = time.time()
    try:
        if '.' in module_name:
            # Handle from X import Y
            parts = module_name.split('.')
            if len(parts) == 2:
                exec(f"from {parts[0]} import {parts[1]}")
            else:
                exec(f"import {module_name}")
        else:
            exec(f"import {module_name}")
        duration = time.time() - start
        print(f"  ✅ SUCCESS ({duration:.3f}s)")
        return True
    except Exception as e:
        duration = time.time() - start
        print(f"  ❌ FAILED ({duration:.3f}s): {e}")
        return False

def main():
    print("🔍 DEBUGGING SERVER.PY IMPORTS")
    print("=" * 50)
    
    # Test basic imports first
    basic_imports = [
        "asyncio",
        "json", 
        "sys",
        "os",
        "subprocess",
        "tempfile",
        "traceback",
        "hashlib",
        "time",
        "typing",
        "aiohttp",
        "logging",
        "pathlib",
        "ast",
        "re",
        "uuid",
        "requests"
    ]
    
    print("\n📦 Testing basic imports...")
    for imp in basic_imports:
        test_import(imp)
    
    # Test local imports that might be problematic
    local_imports = [
        ("observability.metrics", "record_backend_result"),
        ("enhanced_agent_teams", "decide_backend_for_role"),
        ("enhanced_mcp_tools", "merged_tools"),
        ("audit_logger", "ImmutableAuditLogger"),
        ("workflow_composer", "WorkflowComposer"),
        ("bedrock_adapter", "bedrock_adapter"),
    ]
    
    print("\n🏠 Testing local imports...")
    for module, item in local_imports:
        test_import(f"{module}.{item}", f"(from {module})")
    
    # Test core imports
    core_imports = [
        ("core.registry", "ToolRegistry"),
        ("core.context_manager", "ContextManager"),
        ("core.executor", "async_executor"),
    ]
    
    print("\n🎯 Testing core imports...")
    for module, item in core_imports:
        test_import(f"{module}.{item}", f"(from {module})")
    
    # Test handlers
    handler_imports = [
        "handlers.research",
        "handlers.agent_teams", 
        "handlers.memory",
        "handlers.code_tools",
        "handlers.workflow",
        "handlers.audit"
    ]
    
    print("\n🛠️  Testing handler imports...")
    for imp in handler_imports:
        test_import(imp)
    
    print("\n✅ Import testing complete!")

if __name__ == "__main__":
    main()
