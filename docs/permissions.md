# Multi-root Filesystem Permissions (Runtime Grants)

This server supports a flexible, safe-by-default filesystem sandbox:

- ALLOWED_BASE_DIRS: OS-path-separator separated list of roots to allow (e.g., `C:\repo;E:\Projects\generative_flow` on Windows)
- ALLOWED_BASE_DIR: Single allowed root (legacy); still honored
- Current working directory (CWD): always included as a default root

The server validates all file operations via `_safe_path()` and `_safe_directory()` against the allowed roots. If a path is outside, a structured error is raised with code `PATH_NOT_ALLOWED` including the attempted path, current allow-list, and a hint to grant access.

## Runtime Permission Tools
- list_allowed_directories → Returns the current allowed roots
- grant_directory_access {"directory": "..."} → Adds a new allowed root (must exist and be a directory)
- deny_directory_access {"directory": "..."} → Removes a previously allowed root

## Recommended Workflow
1. Start the MCP normally (no special env needed). CWD is allowed by default.
2. When a tool needs access outside current roots, it will raise `PATH_NOT_ALLOWED` with a clear hint.
3. Grant explicit access at runtime using `grant_directory_access`.
4. Optionally revoke with `deny_directory_access` when the task is done.

## Safety Notes
- Permissions are in-memory for the process lifetime (persist only if you externalize via storage).
- All operations remain read/write restricted to the explicit roots.
- Combine with environment permissions and audit tools for defense-in-depth.

