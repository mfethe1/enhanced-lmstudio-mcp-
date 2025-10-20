# MCP Working Directory (cwd) Explained

**Understanding the difference between server location and working directory**

---

## 🎯 **The Key Concept**

When using an MCP server across multiple workspaces, you need to understand **TWO different paths**:

1. **Server Location** - Where the MCP server code lives (FIXED)
2. **Working Directory (cwd)** - Where the MCP server operates (DYNAMIC)

---

## 📍 **Visual Explanation**

```
┌─────────────────────────────────────────────────────────────┐
│  MCP Server Installation (FIXED LOCATION)                   │
│  C:/Users/mfeth/.mcp-servers/lmstudio-mcp/                 │
│  ├── server.py          ← The MCP server code              │
│  ├── handlers/                                              │
│  ├── core/                                                  │
│  └── ...                                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    Runs the server
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Working Directory (DYNAMIC - Changes per workspace)        │
│                                                              │
│  When you open: E:\Projects\ProjectA                        │
│  ├── src/               ← MCP operates here                 │
│  ├── tests/                                                 │
│  └── package.json                                           │
│                                                              │
│  When you open: C:\Users\mfeth\Documents\ProjectB          │
│  ├── app/               ← MCP operates here                 │
│  ├── config/                                                │
│  └── README.md                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ **Configuration Breakdown**

### **Correct Configuration**

```json
{
  "mcpServers": {
    "jarvis": {
      "command": "py",
      "args": [
        "-3",
        "-u",
        "C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py"
      ],
      "cwd": "${workspaceFolder}",
      "env": {
        "ALLOWED_BASE_DIRS": "${workspaceFolder};C:/Users/mfeth/.mcp-servers/lmstudio-mcp;E:/Projects"
      }
    }
  }
}
```

### **What Each Part Does**

| Parameter | Value | Purpose | Changes? |
|-----------|-------|---------|----------|
| `args` | `C:/Users/.../server.py` | **Where to find the MCP server code** | ❌ No (fixed) |
| `cwd` | `${workspaceFolder}` | **Where the MCP server operates** | ✅ Yes (per workspace) |
| `ALLOWED_BASE_DIRS` | `${workspaceFolder};...` | **Which directories MCP can access** | ✅ Yes (per workspace) |

---

## 🔄 **How It Works in Practice**

### **Scenario 1: Working on Web App**

**You open**: `E:\Projects\my-web-app`

**What happens**:
```
Server Location:  C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py  (fixed)
Working Directory: E:\Projects\my-web-app                              (dynamic)
Allowed Dirs:     E:\Projects\my-web-app;                              (dynamic)
                  C:/Users/mfeth/.mcp-servers/lmstudio-mcp;
                  E:/Projects
```

**MCP commands**:
- `list_directory("src")` → Lists `E:\Projects\my-web-app\src`
- `view("package.json")` → Reads `E:\Projects\my-web-app\package.json`
- `save_file("new.js", ...)` → Creates `E:\Projects\my-web-app\new.js`

---

### **Scenario 2: Working on Python Project**

**You open**: `C:\Users\mfeth\Documents\python-project`

**What happens**:
```
Server Location:  C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py  (fixed)
Working Directory: C:\Users\mfeth\Documents\python-project             (dynamic)
Allowed Dirs:     C:\Users\mfeth\Documents\python-project;             (dynamic)
                  C:/Users/mfeth/.mcp-servers/lmstudio-mcp;
                  E:/Projects
```

**MCP commands**:
- `list_directory("tests")` → Lists `C:\Users\mfeth\Documents\python-project\tests`
- `view("main.py")` → Reads `C:\Users\mfeth\Documents\python-project\main.py`
- `save_file("utils.py", ...)` → Creates `C:\Users\mfeth\Documents\python-project\utils.py`

---

## 🎓 **Why This Design?**

### **Server Location is Fixed**
- The MCP server code doesn't move
- You install it once in a central location
- All workspaces use the same server installation
- Updates to the server affect all projects

### **Working Directory is Dynamic**
- Each project has its own directory
- The MCP server needs to operate in the current project
- File operations should be relative to the current project
- Switching workspaces changes the working directory automatically

---

## 🔒 **Security: ALLOWED_BASE_DIRS**

The `ALLOWED_BASE_DIRS` environment variable controls which directories the MCP server can access.

### **Option 1: Workspace Only (Most Restrictive)**
```json
"ALLOWED_BASE_DIRS": "${workspaceFolder}"
```
- MCP can **only** access files in the current workspace
- Most secure
- May be too restrictive if you need to access other directories

### **Option 2: Workspace + Common Directories (Recommended)**
```json
"ALLOWED_BASE_DIRS": "${workspaceFolder};C:/Users/mfeth/.mcp-servers/lmstudio-mcp;E:/Projects"
```
- MCP can access:
  - Current workspace (dynamic)
  - MCP server directory (for reading configs, logs, etc.)
  - Your projects directory (for cross-project operations)
- Good balance of security and flexibility

### **Option 3: Multiple Specific Directories**
```json
"ALLOWED_BASE_DIRS": "${workspaceFolder};C:/Users/mfeth/Documents;E:/Projects;D:/Backups"
```
- MCP can access multiple specific directories
- Use when you need access to specific locations

---

## 🚫 **Common Mistakes**

### **❌ WRONG: Using ${workspaceFolder} for Server Location**
```json
{
  "args": ["-3", "-u", "${workspaceFolder}/server.py"],  // ❌ WRONG
  "cwd": "${workspaceFolder}"
}
```
**Problem**: The server code is NOT in your project workspace!  
**Result**: MCP server fails to start with "file not found"

---

### **❌ WRONG: Hardcoding Working Directory**
```json
{
  "args": ["-3", "-u", "C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py"],
  "cwd": "E:/Projects/ProjectA"  // ❌ WRONG - hardcoded
}
```
**Problem**: When you open ProjectB, MCP still operates in ProjectA!  
**Result**: File operations happen in the wrong directory

---

### **✅ CORRECT: Fixed Server, Dynamic Working Directory**
```json
{
  "args": ["-3", "-u", "C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py"],  // ✅ Fixed
  "cwd": "${workspaceFolder}"  // ✅ Dynamic
}
```
**Result**: Server runs from fixed location, operates in current workspace

---

## 🧪 **Testing Your Configuration**

### **Test 1: Verify Server Starts**
1. Open any workspace in VSCode/Augment
2. Check MCP server logs for startup messages
3. Should see: "MCP server started successfully"

### **Test 2: Verify Working Directory**
1. Open workspace: `E:\Projects\TestProject`
2. Run: `list_directory(".")`
3. Should list contents of `E:\Projects\TestProject`

### **Test 3: Verify Across Workspaces**
1. Open workspace: `E:\Projects\ProjectA`
2. Run: `list_directory(".")`
3. Should list contents of `ProjectA`
4. Close and open workspace: `C:\Users\mfeth\Documents\ProjectB`
5. Run: `list_directory(".")`
6. Should list contents of `ProjectB` (NOT ProjectA!)

---

## 📋 **Quick Reference**

### **What Should Be Fixed?**
- ✅ Server location (`args`)
- ✅ Python command (`command`)

### **What Should Be Dynamic?**
- ✅ Working directory (`cwd`)
- ✅ Allowed directories (includes `${workspaceFolder}`)

### **Template Configuration**
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "py",
      "args": ["-3", "-u", "C:/Users/YOUR_USERNAME/.mcp-servers/lmstudio-mcp/server.py"],
      "cwd": "${workspaceFolder}",
      "env": {
        "ALLOWED_BASE_DIRS": "${workspaceFolder};C:/Users/YOUR_USERNAME/.mcp-servers/lmstudio-mcp;YOUR_PROJECTS_DIR"
      }
    }
  }
}
```

---

## 🎉 **Summary**

**The Golden Rule**:
- **Server location** = Where the code is (FIXED)
- **Working directory** = Where you're working (DYNAMIC)

**Your Configuration**:
```json
"args": ["C:/Users/mfeth/.mcp-servers/lmstudio-mcp/server.py"],  // ✅ FIXED
"cwd": "${workspaceFolder}",                                      // ✅ DYNAMIC
"ALLOWED_BASE_DIRS": "${workspaceFolder};..."                     // ✅ DYNAMIC + EXTRAS
```

**This is CORRECT!** ✅

The MCP server will:
- Always run from the same installation location
- Operate in whichever workspace you have open
- Have access to the current workspace plus any additional directories you specify

**Result**: One MCP server installation works across all your projects! 🚀

