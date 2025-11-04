#!/usr/bin/env python3
"""
Demo script for Amazon Bedrock integration with Strands MCP server.
Shows how to use Claude models via Bedrock for enhanced capabilities.
"""
import os
import sys
import json
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bedrock_adapter import BedrockAdapter
    import server
    BEDROCK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Bedrock adapter not available: {e}")
    print("Install boto3: pip install boto3")
    BEDROCK_AVAILABLE = False
    sys.exit(1)


async def test_bedrock_adapter():
    """Test Bedrock adapter directly."""
    print("🔧 Testing Bedrock Adapter")
    print("=" * 50)
    
    try:
        adapter = BedrockAdapter()
        
        # Check availability
        if not adapter.is_available():
            print("❌ Bedrock not available. Check AWS credentials and region.")
            return False
        
        print("✅ Bedrock client initialized successfully")
        
        # List available models
        models = adapter.get_available_models()
        print(f"📋 Available Anthropic models: {len(models)}")
        for model in models[:3]:  # Show first 3
            print(f"   - {model}")
        
        # Test chat completion
        print("\n💬 Testing chat completion...")
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ]
        
        response = await adapter.achat_completion(
            model="claude-3-5-sonnet-latest",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            print(f"✅ Response: {content}")
            print(f"📊 Usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion tokens")
            return True
        else:
            print("❌ No valid response received")
            return False
            
    except Exception as e:
        print(f"❌ Bedrock test failed: {e}")
        return False


async def test_server_integration():
    """Test Bedrock integration with MCP server."""
    print("\n🔗 Testing Server Integration")
    print("=" * 50)
    
    try:
        # Set environment for Bedrock
        os.environ["USE_BEDROCK"] = "1"
        os.environ["BEDROCK_REGION"] = "us-east-1"
        
        # Create server instance
        mcp_server = server.EnhancedLMStudioMCPServer()
        
        # Test router with Bedrock backend
        print("🎯 Testing router with Bedrock backend...")
        
        response = await mcp_server.route_chat(
            prompt="Explain quantum computing in simple terms. Keep it under 100 words.",
            backend="anthropic",
            temperature=0.5,
            max_tokens=150
        )
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"✅ Router response: {content[:200]}...")
            return True
        else:
            print("❌ No valid response from router")
            return False
            
    except Exception as e:
        print(f"❌ Server integration test failed: {e}")
        return False


def check_aws_credentials():
    """Check if AWS credentials are configured."""
    print("🔐 Checking AWS Credentials")
    print("=" * 50)
    
    # Check environment variables
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("BEDROCK_REGION", "us-east-1")
    
    if access_key and secret_key:
        print("✅ AWS credentials found in environment")
        print(f"📍 Region: {region}")
        return True
    
    # Check AWS credentials file
    aws_creds_file = Path.home() / ".aws" / "credentials"
    aws_config_file = Path.home() / ".aws" / "config"
    
    if aws_creds_file.exists():
        print("✅ AWS credentials file found")
        print(f"📍 Region: {region}")
        return True
    
    print("❌ AWS credentials not found")
    print("Set up credentials using one of these methods:")
    print("1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
    print("2. AWS CLI: aws configure")
    print("3. IAM roles (if running on EC2)")
    return False


async def main():
    """Main demo function."""
    print("🚀 Bedrock Integration Demo")
    print("=" * 50)
    
    if not BEDROCK_AVAILABLE:
        return
    
    # Check prerequisites
    if not check_aws_credentials():
        print("\n⚠️  Configure AWS credentials to continue")
        return
    
    # Test Bedrock adapter
    adapter_success = await test_bedrock_adapter()
    
    # Test server integration
    if adapter_success:
        server_success = await test_server_integration()
        
        if server_success:
            print("\n🎉 All tests passed!")
            print("\n📝 To enable Bedrock in production:")
            print("1. Set USE_BEDROCK=1 in your environment")
            print("2. Set BEDROCK_REGION=us-east-1 (or your preferred region)")
            print("3. Ensure AWS credentials are configured")
            print("4. Update model names in ANTHROPIC_MODEL_* variables to use Bedrock IDs")
        else:
            print("\n⚠️  Server integration had issues")
    else:
        print("\n⚠️  Bedrock adapter test failed")


if __name__ == "__main__":
    asyncio.run(main())
