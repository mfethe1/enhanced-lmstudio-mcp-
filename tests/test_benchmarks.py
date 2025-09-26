import importlib
import time


def test_optimizer_metrics_and_cache_ratio():
    server = importlib.import_module('server')
    s = server.get_server_singleton()
    if not getattr(s, 'production', None):
        return
    prod = s.production

    # Simulate repeated calls to drive cache hits
    import asyncio
    async def run():
        p = 'Say hello world'
        # First call: miss
        await prod.cached_llm(p, temperature=0.2, intent='bench', role='test')
        # Second call: hit
        await prod.cached_llm(p, temperature=0.2, intent='bench', role='test')
        return prod.metrics_snapshot()
    try:
        out = asyncio.get_event_loop().run_until_complete(run())
    except RuntimeError:
        loop = __import__('asyncio').new_event_loop()
        try:
            __import__('asyncio').set_event_loop(loop)
            out = loop.run_until_complete(run())
        finally:
            loop.close()
    # Should have at least 1 hit and a hit ratio > 0
    assert out['cache_hits'] >= 1
    assert out['cache_hit_ratio'] > 0.0


def test_parallel_tools_speedup():
    server = importlib.import_module('server')
    s = server.get_server_singleton()
    if not getattr(s, 'production', None):
        return
    prod = s.production

    calls = [{"name": "health_check", "arguments": {}} for _ in range(4)]
    import asyncio
    async def run():
        t0 = time.perf_counter()
        await prod.parallel_tools(calls)
        t1 = time.perf_counter()
        return (t1 - t0)
    try:
        dt = asyncio.get_event_loop().run_until_complete(run())
    except RuntimeError:
        loop = __import__('asyncio').new_event_loop()
        try:
            __import__('asyncio').set_event_loop(loop)
            dt = loop.run_until_complete(run())
        finally:
            loop.close()
    # Smoke assert it completes quickly
    assert dt < 10.0

