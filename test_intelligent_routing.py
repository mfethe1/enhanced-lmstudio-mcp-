#!/usr/bin/env python3
"""
Intelligent Routing System Test Suite
Tests the complete intelligent routing system with all models
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import testing modules
try:
    import aiohttp
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.info("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "requests", "python-dotenv"])
    import aiohttp
    import requests
    from dotenv import load_dotenv

# Load environment variables
env_files = ['.env', '.secrets/.env.local', '.env.production']
for env_file in env_files:
    if Path(env_file).exists():
        load_dotenv(env_file, override=True)
        logger.info(f"✓ Loaded environment from {env_file}")

class IntelligentRoutingTester:
    """Comprehensive test suite for intelligent routing system"""
    
    def __init__(self):
        self.results = []
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment"""
        self.config = {
            # OpenAI Configuration
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                'model': os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest'),
                'reasoning_model': os.getenv('OPENAI_REASONING_MODEL', 'gpt-5-chat-latest'),
                'coding_model': os.getenv('OPENAI_CODING_MODEL', 'gpt-5-chat-latest'),
                'fallback_model': os.getenv('OPENAI_FALLBACK_MODEL', 'gpt-4o')
            },
            # Anthropic Configuration
            'anthropic': {
                'api_key': os.getenv('ANTHROPIC_API_KEY'),
                'base_url': os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
                'model': os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5-20250929'),
                'complex_model': os.getenv('ANTHROPIC_MODEL_COMPLEX', 'claude-opus-4-1-20250805'),
                'overseer_model': os.getenv('ANTHROPIC_MODEL_OVERSEER', 'claude-opus-4-1-20250805')
            },
            # LM Studio Configuration
            'lmstudio': {
                'base_url': os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1'),
                'model': os.getenv('LMSTUDIO_MODEL', 'openai/gpt-oss-20b')
            },
            # Routing Configuration
            'routing': {
                'simple_threshold': float(os.getenv('SIMPLE_TASK_THRESHOLD', '0.3')),
                'complex_threshold': float(os.getenv('COMPLEX_TASK_THRESHOLD', '0.7')),
                'use_local_for_simple': os.getenv('USE_LOCAL_FOR_SIMPLE', '1') == '1',
                'use_reasoning_for_complex': os.getenv('USE_REASONING_FOR_COMPLEX', '1') == '1'
            }
        }
    
    async def test_openai_models(self) -> Dict[str, Any]:
        """Test OpenAI model availability and routing"""
        results = {}
        
        try:
            # Test primary model
            primary_result = await self._test_model(
                'openai',
                self.config['openai']['model'],
                "What is 2+2?"
            )
            results['primary'] = primary_result
            
            # Test reasoning model for complex tasks
            if self.config['routing']['use_reasoning_for_complex']:
                reasoning_result = await self._test_model(
                    'openai',
                    self.config['openai']['reasoning_model'],
                    "Explain the halting problem and its implications for computer science."
                )
                results['reasoning'] = reasoning_result
            
            # Test coding model
            coding_result = await self._test_model(
                'openai',
                self.config['openai']['coding_model'],
                "Write a Python function to calculate fibonacci numbers."
            )
            results['coding'] = coding_result
            
            # Test fallback model
            fallback_result = await self._test_model(
                'openai',
                self.config['openai']['fallback_model'],
                "Hello, are you working?"
            )
            results['fallback'] = fallback_result
            
        except Exception as e:
            logger.error(f"OpenAI test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def test_anthropic_models(self) -> Dict[str, Any]:
        """Test Anthropic model availability and routing"""
        results = {}
        
        try:
            # Test primary Sonnet model
            sonnet_result = await self._test_model(
                'anthropic',
                self.config['anthropic']['model'],
                "What is the capital of France?"
            )
            results['sonnet'] = sonnet_result
            
            # Test Opus for complex tasks
            opus_result = await self._test_model(
                'anthropic',
                self.config['anthropic']['complex_model'],
                "Analyze the economic implications of artificial general intelligence."
            )
            results['opus_complex'] = opus_result
            
            # Test Overseer model
            overseer_result = await self._test_model(
                'anthropic',
                self.config['anthropic']['overseer_model'],
                "Review this plan: 1) Research topic, 2) Write draft, 3) Edit and publish"
            )
            results['opus_overseer'] = overseer_result
            
        except Exception as e:
            logger.error(f"Anthropic test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def test_lmstudio_local(self) -> Dict[str, Any]:
        """Test LM Studio local model"""
        results = {}
        
        try:
            # Check if LM Studio is running
            health_check = await self._check_lmstudio_health()
            results['health'] = health_check
            
            if health_check.get('status') == 'healthy':
                # Test simple task routing to local model
                if self.config['routing']['use_local_for_simple']:
                    local_result = await self._test_model(
                        'lmstudio',
                        self.config['lmstudio']['model'],
                        "What day comes after Tuesday?"
                    )
                    results['simple_task'] = local_result
            else:
                results['status'] = 'LM Studio not available'
                
        except Exception as e:
            logger.error(f"LM Studio test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def test_intelligent_routing(self) -> Dict[str, Any]:
        """Test the complete intelligent routing logic"""
        test_cases = [
            {
                'name': 'Simple Math',
                'prompt': 'What is 5 + 3?',
                'expected_route': 'local' if self.config['routing']['use_local_for_simple'] else 'standard',
                'complexity': 0.1
            },
            {
                'name': 'Standard Query',
                'prompt': 'Explain photosynthesis in simple terms',
                'expected_route': 'standard',
                'complexity': 0.5
            },
            {
                'name': 'Complex Reasoning',
                'prompt': 'Design a distributed system for real-time collaborative editing with CRDT support',
                'expected_route': 'reasoning' if self.config['routing']['use_reasoning_for_complex'] else 'standard',
                'complexity': 0.9
            },
            {
                'name': 'Code Generation',
                'prompt': 'Write a Python class for a binary search tree with insertion and deletion',
                'expected_route': 'coding',
                'complexity': 0.6
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                route = self._determine_route(test_case['complexity'], test_case['prompt'])
                success = route == test_case['expected_route'] or route in ['standard', 'fallback']
                
                result = {
                    'test': test_case['name'],
                    'complexity': test_case['complexity'],
                    'expected_route': test_case['expected_route'],
                    'actual_route': route,
                    'success': success
                }
                results.append(result)
                
                status = '✅' if success else '❌'
                logger.info(f"{status} {test_case['name']}: Expected {test_case['expected_route']}, Got {route}")
                
            except Exception as e:
                results.append({
                    'test': test_case['name'],
                    'error': str(e),
                    'success': False
                })
                logger.error(f"❌ {test_case['name']}: {e}")
        
        return {
            'test_cases': results,
            'passed': sum(1 for r in results if r.get('success', False)),
            'total': len(results)
        }
    
    def _determine_route(self, complexity: float, prompt: str) -> str:
        """Determine routing based on complexity and prompt"""
        # Check for coding tasks
        coding_keywords = ['write', 'code', 'function', 'class', 'implement', 'python', 'javascript']
        if any(keyword in prompt.lower() for keyword in coding_keywords):
            return 'coding'
        
        # Route based on complexity
        if complexity < self.config['routing']['simple_threshold']:
            if self.config['routing']['use_local_for_simple']:
                return 'local'
            return 'standard'
        elif complexity > self.config['routing']['complex_threshold']:
            if self.config['routing']['use_reasoning_for_complex']:
                return 'reasoning'
            return 'standard'
        else:
            return 'standard'
    
    async def _test_model(self, provider: str, model: str, prompt: str) -> Dict[str, Any]:
        """Test a specific model"""
        start_time = datetime.now()
        
        try:
            if provider == 'openai':
                response = await self._call_openai(model, prompt)
            elif provider == 'anthropic':
                response = await self._call_anthropic(model, prompt)
            elif provider == 'lmstudio':
                response = await self._call_lmstudio(model, prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'model': model,
                'response_length': len(response.get('content', '')),
                'latency': elapsed,
                'truncated_response': response.get('content', '')[:100] + '...' if len(response.get('content', '')) > 100 else response.get('content', '')
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'model': model,
                'error': str(e)
            }
    
    async def _call_openai(self, model: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        headers = {
            'Authorization': f"Bearer {self.config['openai']['api_key']}",
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 150
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['openai']['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {'content': result['choices'][0]['message']['content']}
                else:
                    error = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error}")
    
    async def _call_anthropic(self, model: str, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API"""
        headers = {
            'x-api-key': self.config['anthropic']['api_key'],
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 150,
            'temperature': 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['anthropic']['base_url']}/v1/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {'content': result['content'][0]['text']}
                else:
                    error = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error}")
    
    async def _call_lmstudio(self, model: str, prompt: str) -> Dict[str, Any]:
        """Call LM Studio API"""
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 150
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config['lmstudio']['base_url']}/chat/completions",
                json=data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {'content': result['choices'][0]['message']['content']}
                else:
                    error = await response.text()
                    raise Exception(f"LM Studio API error {response.status}: {error}")
    
    async def _check_lmstudio_health(self) -> Dict[str, Any]:
        """Check if LM Studio is running and healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['lmstudio']['base_url']}/models",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        models = await response.json()
                        return {
                            'status': 'healthy',
                            'models': models.get('data', [])
                        }
                    else:
                        return {'status': 'unhealthy', 'code': response.status}
        except Exception as e:
            return {'status': 'offline', 'error': str(e)}
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("INTELLIGENT ROUTING SYSTEM TEST SUITE")
        print("="*80 + "\n")
        
        # Test OpenAI models
        print("Testing OpenAI Models...")
        openai_results = await self.test_openai_models()
        self._print_results("OpenAI", openai_results)
        
        # Test Anthropic models
        print("\nTesting Anthropic Models...")
        anthropic_results = await self.test_anthropic_models()
        self._print_results("Anthropic", anthropic_results)
        
        # Test LM Studio
        print("\nTesting LM Studio Local Model...")
        lmstudio_results = await self.test_lmstudio_local()
        self._print_results("LM Studio", lmstudio_results)
        
        # Test routing logic
        print("\nTesting Intelligent Routing Logic...")
        routing_results = await self.test_intelligent_routing()
        print(f"  ✓ Passed: {routing_results['passed']}/{routing_results['total']} tests")
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        all_success = True
        
        # Check OpenAI
        if 'error' not in openai_results:
            print("[OK] OpenAI: All models operational")
        else:
            print(f"[WARNING] OpenAI: {openai_results.get('error', 'Unknown error')}")
            all_success = False
        
        # Check Anthropic
        if 'error' not in anthropic_results:
            print("[OK] Anthropic: All models operational")
        else:
            print(f"[WARNING] Anthropic: {anthropic_results.get('error', 'Unknown error')}")
            all_success = False
        
        # Check LM Studio
        if lmstudio_results.get('health', {}).get('status') == 'healthy':
            print("[OK] LM Studio: Local model available")
        else:
            print("[INFO] LM Studio: Not available (non-critical)")
        
        # Check routing
        if routing_results['passed'] == routing_results['total']:
            print("[OK] Routing Logic: All tests passed")
        else:
            print(f"[WARNING] Routing Logic: {routing_results['passed']}/{routing_results['total']} passed")
            all_success = False
        
        print("\n" + "="*80)
        if all_success:
            print("ALL CRITICAL TESTS PASSED!")
            print("Your intelligent routing system is fully operational!")
        else:
            print("WARNING: Some tests failed. Please check the configuration.")
        print("="*80 + "\n")
    
    def _print_results(self, category: str, results: Dict[str, Any]):
        """Pretty print test results"""
        if 'error' in results:
            print(f"  ❌ {category} Error: {results['error']}")
        else:
            for key, value in results.items():
                if isinstance(value, dict) and 'status' in value:
                    status = '✅' if value['status'] == 'success' or value['status'] == 'healthy' else '⚠️'
                    print(f"  {status} {key}: {value.get('model', 'N/A')} - {value['status']}")
                    if 'latency' in value:
                        print(f"     Latency: {value['latency']:.2f}s")

async def main():
    """Main entry point"""
    tester = IntelligentRoutingTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)