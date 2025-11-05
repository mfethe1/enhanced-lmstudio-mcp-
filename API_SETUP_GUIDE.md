# API Setup Guide - Complete Instructions

**Status**: 🟢 LM Studio Working | 🔴 OpenAI Quota Exceeded | 🔴 Anthropic Model Name Issue

## 📊 Current Situation

Your diagnostic results show:
- ✅ **LM Studio**: Working perfectly with 15 models loaded
- ❌ **OpenAI**: API key valid, but quota exceeded (need to add billing)
- ❌ **Anthropic**: API key may be valid, but wrong model name configured

## 🎯 Quick Fix Path

### Option 1: Use LM Studio Only (Immediate Solution - Already Working!)
Your LM Studio is working perfectly. The server will automatically use it when OpenAI/Anthropic fail.

**No action needed** - your server is functional with local models.

### Option 2: Fix OpenAI API (Recommended)

#### Problem Identified:
- ✅ API Key is valid (can list models)
- ❌ Quota exceeded (no billing/credits)
- ⚠️  Wrong model name: `gpt-5` doesn't exist yet

#### Step-by-Step Fix:

1. **Set up OpenAI Billing** (Required)
   ```
   → Go to: https://platform.openai.com/account/billing
   → Click "Add payment method"
   → Add a credit card
   → Add at least $5-10 in credits
   ```

2. **Update Model Name in Configuration**
   
   Open `.secrets/.env.local` and change:
   ```bash
   # FROM (incorrect):
   OPENAI_MODEL=gpt-5
   
   # TO (correct):
   OPENAI_MODEL=gpt-4o-mini
   ```

3. **Alternative: Get a Fresh API Key**
   
   If billing doesn't work, try a new key:
   ```
   → Go to: https://platform.openai.com/api-keys
   → Click "Create new secret key"
   → Name it: "MCP Server"
   → Copy the key
   → Update OPENAI_API_KEY in .secrets/.env.local
   ```

#### Recommended OpenAI Models (in order of cost):
```bash
gpt-4o-mini          # Cheapest, fast, very capable
gpt-4o               # More capable, medium cost  
gpt-4-turbo          # Most capable, higher cost
o1-mini              # Reasoning model (special use)
```

### Option 3: Fix Anthropic API

#### Problem Identified:
- ⚠️  Wrong model name: `claude-3-5-sonnet-latest`
- Should be: `claude-3-5-sonnet-20241022` (with date suffix)

#### Step-by-Step Fix:

1. **Update Model Name**
   
   Open `.secrets/.env.local` and add/update:
   ```bash
   ANTHROPIC_API_KEY={{ANTHROPIC_API_KEY}}
   ANTHROPIC_BASE_URL=https://api.anthropic.com
   ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
   ANTHROPIC_MODEL_COMPLEX=claude-3-opus-20240229
   ANTHROPIC_MODEL_OVERSEER=claude-3-opus-20240229
   ```

2. **Verify Billing is Active**
   ```
   → Go to: https://console.anthropic.com/settings/billing
   → Ensure you have a payment method and credits
   ```

3. **Test the API Key**
   ```
   → Go to: https://console.anthropic.com/settings/keys
   → Verify the key is active
   → If needed, create a new key
   ```

#### Valid Anthropic Model Names:
```bash
claude-3-5-sonnet-20241022    # Latest Sonnet (recommended)
claude-3-opus-20240229        # Most capable
claude-3-haiku-20240307       # Fastest, cheapest
claude-3-sonnet-20240229      # Balanced (older)
```

## 🔧 Complete Fixed Configuration

Here's the complete `.secrets/.env.local` file you should have:

```bash
# LM Studio (OpenAI-compatible) - Local AI models
LMSTUDIO_API_BASE=http://localhost:1234/v1
LMSTUDIO_API_KEY=sk-noauth
LMSTUDIO_MODEL=openai/gpt-oss-20b

# OpenAI API - Cloud service
OPENAI_API_KEY={{OPENAI_API_KEY}}
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo

# Anthropic API - Claude models  
ANTHROPIC_API_KEY={{ANTHROPIC_API_KEY}}
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MODEL_COMPLEX=claude-3-opus-20240229
ANTHROPIC_MODEL_OVERSEER=claude-3-opus-20240229

# Firecrawl API - Web research
FIRECRAWL_API_KEY={{FIRECRAWL_API_KEY}}

# Routing configuration
ANTHROPIC_SMART_SWITCH=1
OPUS_FOR_ANALYSIS=1
OPUS_FOR_LOW_CONF=1
LOW_CONF_THRESHOLD=0.5

# Performance settings
HTTP_CONNECT_TIMEOUT=10
HTTP_READ_TIMEOUT_SIMPLE=60
HTTP_READ_TIMEOUT_COMPLEX=180
PROACTIVE_RESEARCH_ENABLED=0
CIRCUIT_BREAKER_ENABLED=1
```

## ✅ Testing Your Setup

After making changes, test with:

```bash
# Run the diagnostic tool again
python diagnose_api_connections.py

# Or test the MCP server
python test_agentic_functionality.py
```

## 🎓 Understanding the Issues

### Why OpenAI Failed:
1. **Quota Exceeded**: OpenAI requires billing to be set up
2. **Model Name**: `gpt-5` doesn't exist (use `gpt-4o-mini`)
3. **Solution**: Add payment method + fix model name

### Why Anthropic Failed:
1. **Model Name Format**: Anthropic requires date suffixes
2. **Wrong**: `claude-3-5-sonnet-latest`
3. **Correct**: `claude-3-5-sonnet-20241022`

### Why LM Studio Works:
- It's running locally on your machine
- No API keys or billing needed
- Already has 15 models loaded

## 💡 Best Practice Recommendations

### For Cost-Effectiveness:
```bash
Primary: LM Studio (free, local)
Fallback: OpenAI gpt-4o-mini (cheap, cloud)
Complex tasks: Anthropic claude-3-5-sonnet-20241022
```

### For Maximum Capability:
```bash
Primary: Anthropic claude-3-5-sonnet-20241022
Fallback: OpenAI gpt-4o
Local: LM Studio for privacy-sensitive tasks
```

### For Development (Current Setup):
```bash
Primary: LM Studio ✅ (Working)
Fallback: OpenAI (Fix billing + model name)
Advanced: Anthropic (Fix model name)
```

## 🚀 Quick Actions (Priority Order)

### Immediate (5 minutes):
1. ✅ You're already functional with LM Studio
2. Update Anthropic model name to `claude-3-5-sonnet-20241022`
3. Update OpenAI model name to `gpt-4o-mini`

### Short-term (15 minutes):
1. Set up OpenAI billing at https://platform.openai.com/account/billing
2. Add $10-20 in credits
3. Test with: `python diagnose_api_connections.py`

### Optional:
1. Verify Anthropic billing is active
2. Generate fresh API keys if needed
3. Test all APIs working together

## 📞 Support Resources

- **OpenAI**: https://help.openai.com/
- **Anthropic**: https://support.anthropic.com/
- **LM Studio**: https://lmstudio.ai/docs

## 🎯 Current Status Summary

```
✅ System is OPERATIONAL with LM Studio
⚠️  OpenAI needs billing + model name fix
⚠️  Anthropic needs model name fix

Action Required:
1. Fix model names (2 minutes)
2. Add OpenAI billing (5 minutes)
3. Test with diagnostic tool
```

---

**After making changes, run:**
```bash
python diagnose_api_connections.py
```

This will verify all APIs are working correctly!