# MCP Path Configuration Guide

**How to automatically load the working directory into MCP configuration**

This guide explains multiple approaches to avoid hardcoding full directory paths in your `mcp.json` configuration.

---

## 🎯 **Recommended Approach: Use `${workspaceFolder}` Variable**

### **What is `${workspaceFolder}`?**

`${workspaceFolder}` is a built-in variable supported by VSCode, Augment, and most MCP clients that automatically resolves to the currently open workspace directory.

### **Configuration Example**

```json
{
  "description": "Jarvis MCP Server Configuration - Uses ${workspaceFolder} for automatic path resolution",
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "py",
      "args": ["-3", "-u", "${workspaceFolder}/server.py"],
      "cwd": "${workspaceFolder}",
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here",
        // ... other environment variables
      }
    }
  }
}
```

### **Benefits**

✅ **Automatic**: No need to update paths when moving projects  
✅ **Portable**: Works on any machine without modification  
✅ **Clean**: No hardcoded paths in configuration  
✅ **Standard**: Supported by VSCode, Augment, and most MCP clients

### **How It Works**

1. You open a workspace in VSCode/Augment (e.g., `C:/Users/mfeth/.mcp-servers/lmstudio-mcp`)
2. The MCP client automatically replaces `${workspaceFolder}` with the actual path
3. The MCP server starts with the correct working directory

---

## 🔧 **Alternative Approaches**

### **Option 2: Use Environment Variables**

You can use environment variables in your `mcp.json`:

```json
{
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "py",
      "args": ["-3", "-u", "${env:JARVIS_MCP_PATH}/server.py"],
      "cwd": "${env:JARVIS_MCP_PATH}",
      "env": {
        "JARVIS_MCP_PATH": "C:/Users/mfeth/.mcp-servers/lmstudio-mcp"
      }
    }
  }
}
```

**Benefits**:
- Can be set system-wide
- Easy to change without editing JSON

**Drawbacks**:
- Still requires setting the environment variable
- Less portable than `${workspaceFolder}`

---

### **Option 3: Use Home Directory Variable**

Some MCP clients support `${userHome}` or `~`:

```json
{
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "py",
      "args": ["-3", "-u", "${userHome}/.mcp-servers/lmstudio-mcp/server.py"],
      "cwd": "${userHome}/.mcp-servers/lmstudio-mcp",
      "env": {}
    }
  }
}
```

**Benefits**:
- Works across different users
- Good for user-specific installations

**Drawbacks**:
- Assumes MCP server is in a fixed location relative to home directory
- Not as flexible as `${workspaceFolder}`

---

### **Option 4: Relative Paths (Limited Support)**

Some MCP clients may support relative paths:

```json
{
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "py",
      "args": ["-3", "-u", "./server.py"],
      "cwd": ".",
      "env": {}
    }
  }
}
```

**Benefits**:
- Very simple

**Drawbacks**:
- ⚠️ **Not widely supported** - depends on where the MCP client runs from
- May not work reliably across different MCP clients

---

## 📋 **Comparison Table**

| Approach | Portability | Ease of Use | Reliability | Best For |
|----------|-------------|-------------|-------------|----------|
| `${workspaceFolder}` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Most cases** |
| Environment Variables | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | System-wide config |
| `${userHome}` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | User-specific installs |
| Relative Paths | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Limited use cases |
| Hardcoded Paths | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Single machine only |

---

## 🚀 **Quick Start: Migrate to `${workspaceFolder}`**

### **Step 1: Update Your `mcp.json`**

Replace hardcoded paths with `${workspaceFolder}`:

**Before**:
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "py",
      "args": ["-3", "-u", "C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py"],
      "cwd": "C:/Users/mfeth/.mcp-servers/lmstudio-mcp"
    }
  }
}
```

**After**:
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "py",
      "args": ["-3", "-u", "${workspaceFolder}/server.py"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

### **Step 2: Verify Your Workspace**

Make sure you have the MCP server directory open as your workspace in VSCode/Augment:

1. Open VSCode/Augment
2. File → Open Folder
3. Select `C:/Users/mfeth/.mcp-servers/lmstudio-mcp` (or wherever your MCP server is)
4. The workspace folder should now be the MCP server directory

### **Step 3: Restart MCP Client**

1. Restart VSCode/Augment to reload the MCP configuration
2. The MCP server should start with the correct working directory

### **Step 4: Test**

Try a file operation to verify it works:

```python
# This should now work without path errors
list_directory("scripts")
```

---

## 🔍 **Troubleshooting**

### **Problem: `${workspaceFolder}` Not Resolving**

**Symptoms**: MCP server fails to start, or paths are literally `${workspaceFolder}/server.py`

**Solutions**:
1. **Check MCP Client Support**: Ensure your MCP client supports variable substitution
2. **Verify Workspace**: Make sure you have a workspace open (not just individual files)
3. **Check Syntax**: Ensure the variable is exactly `${workspaceFolder}` (case-sensitive)
4. **Restart Client**: Restart VSCode/Augment after changing configuration

### **Problem: Wrong Directory**

**Symptoms**: MCP server starts but file operations fail

**Solutions**:
1. **Check Workspace Root**: Ensure the workspace root is the MCP server directory
2. **Verify `cwd`**: Make sure `"cwd": "${workspaceFolder}"` is set
3. **Check Logs**: Look for "current working directory" in MCP server logs

### **Problem: Works on One Machine, Not Another**

**Symptoms**: Configuration works on your machine but not on a colleague's

**Solutions**:
1. **Use `${workspaceFolder}`**: This is the most portable approach
2. **Avoid Hardcoded Paths**: Never use absolute paths like `C:/Users/...`
3. **Document Requirements**: Make sure everyone opens the same workspace folder

---

## 📝 **Best Practices**

### **DO**

✅ Use `${workspaceFolder}` for maximum portability  
✅ Set `"cwd": "${workspaceFolder}"` to ensure correct working directory  
✅ Document which folder should be opened as the workspace  
✅ Test on a fresh machine to verify portability

### **DON'T**

❌ Hardcode absolute paths like `C:/Users/mfeth/...`  
❌ Assume the MCP client runs from a specific directory  
❌ Mix different path styles (use one approach consistently)  
❌ Forget to set the `cwd` parameter

---

## 🎓 **Advanced: Multiple MCP Servers**

If you have multiple MCP servers in different directories, you can use `${workspaceFolder}` for each:

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "py",
      "args": ["-3", "-u", "${workspaceFolder}/server.py"],
      "cwd": "${workspaceFolder}"
    },
    "other-mcp": {
      "command": "node",
      "args": ["${workspaceFolder}/../other-mcp/index.js"],
      "cwd": "${workspaceFolder}/../other-mcp"
    }
  }
}
```

**Note**: You can use relative paths from `${workspaceFolder}` to reference other directories.

---

## 📚 **Additional Resources**

- **VSCode Variables Reference**: https://code.visualstudio.com/docs/editor/variables-reference
- **MCP Specification**: https://modelcontextprotocol.io/
- **Augment Documentation**: Check Augment's docs for supported variables

---

## 🎉 **Summary**

**Recommended Configuration**:

```json
{
  "description": "Jarvis MCP Server - Automatic path resolution",
  "mcpServers": {
    "jarvis": {
      "type": "stdio",
      "command": "py",
      "args": ["-3", "-u", "${workspaceFolder}/server.py"],
      "cwd": "${workspaceFolder}",
      "env": {
        // Your environment variables here
      }
    }
  }
}
```

**Key Points**:
- ✅ Use `${workspaceFolder}` for automatic path resolution
- ✅ Set `"cwd": "${workspaceFolder}"` for correct working directory
- ✅ Open the MCP server directory as your workspace
- ✅ Restart VSCode/Augment after configuration changes

**Result**: Your MCP configuration will work on any machine without modification! 🚀

