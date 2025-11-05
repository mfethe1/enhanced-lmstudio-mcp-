# Firecrawl MCP Integration

To enable Firecrawl-backed research in Jarvis tools (deep_research, web_search when enabled):

## Best-practice secret storage
- Store FIRECRAWL_API_KEY outside the repo, e.g., in `.secrets/.env.local` (ignored by git).
- Ensure your MCP launcher (IDE or CLI) loads that environment file.
- Do not hardcode tokens in committed JSON; reference the env var instead.

## Recommended config snippet
Use `recommendations/mcp.firecrawl.example.json` as a template and reference `${FIRECRAWL_API_KEY}`:

```json
{
  "mcpServers": {
    "firecrawl-mcp": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"
      }
    }
  }
}
```

Place your actual key in the environment, not in the repo. Example PowerShell session:

```powershell
$env:FIRECRAWL_API_KEY = "fc-..."
# or load from .secrets/.env.local via your launcher
```

## Why env > inline
- Avoids accidental commit/rotation issues
- Works across machines and CI without editing JSON
- Our repo automation never overwrites your local environment

## Troubleshooting
- If deep_research shows "Firecrawl MCP unavailable" ensure:
  1) The firecrawl-mcp server is configured in your MCP runner
  2) FIRECRAWL_API_KEY is present in that process's environment
  3) Network egress is allowed

