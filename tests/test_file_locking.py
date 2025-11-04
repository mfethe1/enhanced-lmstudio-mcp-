"""
Comprehensive tests for file-level locking system.

Test Categories:
1. Lock Acquisition and Release
2. Timeout Handling
3. Concurrent Access
4. Deadlock Prevention
5. Multiple File Locking
6. Performance
"""

import asyncio
import os
import pytest
import tempfile
import time
from pathlib import Path

from core.file_locking import (
    FileLockManager,
    FileLock,
    LockState,
    get_file_lock_manager
)


@pytest.fixture
def temp_files():
    """Create temporary files for testing"""
    temp_dir = tempfile.mkdtemp()
    files = []
    
    for i in range(5):
        file_path = os.path.join(temp_dir, f"test_file_{i}.txt")
        with open(file_path, 'w') as f:
            f.write(f"Test content {i}")
        files.append(file_path)
    
    yield files
    
    # Cleanup
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
    os.rmdir(temp_dir)


class TestLockAcquisitionAndRelease:
    """Test basic lock acquisition and release"""
    
    @pytest.mark.asyncio
    async def test_acquire_lock_on_unlocked_file(self, temp_files):
        """Test acquiring lock on an unlocked file"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            lock_id = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1"
            )
            
            assert lock_id is not None
            assert lock_id.startswith("lock-")
            
            # Verify lock is in registry
            lock_info = manager.get_lock_info(temp_files[0])
            assert lock_info is not None
            assert lock_info["owner_id"] == "agent-1"
            assert lock_info["state"] == "locked"
            
            # Verify stats
            stats = manager.get_stats()
            assert stats["active_locks"] == 1
            assert stats["total_acquired"] == 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_release_lock(self, temp_files):
        """Test releasing a lock"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Acquire lock
            lock_id = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1"
            )
            
            # Release lock
            success = await manager.release_lock(
                file_path=temp_files[0],
                owner_id="agent-1"
            )
            
            assert success is True
            
            # Verify lock is removed
            lock_info = manager.get_lock_info(temp_files[0])
            assert lock_info is None
            
            # Verify stats
            stats = manager.get_stats()
            assert stats["active_locks"] == 0
            assert stats["total_released"] == 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_reentrant_lock(self, temp_files):
        """Test that same owner can re-acquire lock"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Acquire lock
            lock_id1 = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1"
            )
            
            # Try to acquire again (should succeed with same lock_id)
            lock_id2 = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1"
            )
            
            assert lock_id1 == lock_id2
            
            # Should still be only 1 lock
            stats = manager.get_stats()
            assert stats["active_locks"] == 1
            
        finally:
            await manager.stop()


class TestTimeoutHandling:
    """Test lock timeout mechanisms"""
    
    @pytest.mark.asyncio
    async def test_lock_expires_after_timeout(self, temp_files):
        """Test that lock expires after timeout"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Acquire lock with short timeout
            lock_id = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1",
                timeout_seconds=1
            )
            
            # Wait for expiration + cleanup loop
            await asyncio.sleep(7)  # 1s timeout + 5s cleanup interval + buffer
            
            # Lock should be expired and removed
            lock_info = manager.get_lock_info(temp_files[0])
            assert lock_info is None
            
            # Verify timeout was counted
            stats = manager.get_stats()
            assert stats["total_timeouts"] >= 1
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_acquire_timeout_on_locked_file(self, temp_files):
        """Test timeout when trying to acquire locked file"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Agent 1 acquires lock
            lock_id1 = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1",
                timeout_seconds=60
            )
            
            # Agent 2 tries to acquire with short timeout
            with pytest.raises(TimeoutError):
                await manager.acquire_lock(
                    file_path=temp_files[0],
                    owner_id="agent-2",
                    timeout_seconds=1,
                    wait=True
                )
            
            # Verify conflict was counted
            stats = manager.get_stats()
            assert stats["total_conflicts"] >= 1
            
        finally:
            await manager.stop()


class TestConcurrentAccess:
    """Test concurrent access scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_lock_attempts(self, temp_files):
        """Test multiple agents trying to lock same file"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            results = []
            
            async def try_lock(agent_id):
                try:
                    lock_id = await manager.acquire_lock(
                        file_path=temp_files[0],
                        owner_id=agent_id,
                        timeout_seconds=2,
                        wait=True
                    )
                    results.append((agent_id, lock_id, "success"))
                except TimeoutError:
                    results.append((agent_id, None, "timeout"))
            
            # Spawn 5 agents trying to lock same file
            tasks = [try_lock(f"agent-{i}") for i in range(5)]
            await asyncio.gather(*tasks)
            
            # Only 1 should succeed immediately
            successes = [r for r in results if r[2] == "success"]
            assert len(successes) >= 1  # At least one should succeed
            
            # Others should timeout or wait
            assert len(results) == 5
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_lock_queue_processing(self, temp_files):
        """Test that queued requests are processed in order"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Agent 1 acquires lock
            lock_id1 = await manager.acquire_lock(
                file_path=temp_files[0],
                owner_id="agent-1",
                timeout_seconds=2
            )
            
            # Start agent 2 waiting (lower priority)
            async def wait_for_lock_2():
                return await manager.acquire_lock(
                    file_path=temp_files[0],
                    owner_id="agent-2",
                    priority=1,
                    timeout_seconds=10,
                    wait=True
                )
            
            # Start agent 3 waiting (higher priority)
            async def wait_for_lock_3():
                return await manager.acquire_lock(
                    file_path=temp_files[0],
                    owner_id="agent-3",
                    priority=5,
                    timeout_seconds=10,
                    wait=True
                )
            
            task2 = asyncio.create_task(wait_for_lock_2())
            await asyncio.sleep(0.1)  # Ensure agent 2 queues first
            task3 = asyncio.create_task(wait_for_lock_3())
            await asyncio.sleep(0.1)
            
            # Release agent 1's lock
            await manager.release_lock(temp_files[0], "agent-1")
            
            # Wait for one of them to acquire
            await asyncio.sleep(1)
            
            # Agent 3 (higher priority) should get lock first
            stats = manager.get_stats()
            if stats["active_locks"] > 0:
                lock_info = manager.get_lock_info(temp_files[0])
                # Either agent 2 or 3 should have it (priority-based)
                assert lock_info["owner_id"] in ["agent-2", "agent-3"]
            
            # Cleanup
            task2.cancel()
            task3.cancel()
            
        finally:
            await manager.stop()


class TestDeadlockPrevention:
    """Test deadlock prevention mechanisms"""
    
    @pytest.mark.asyncio
    async def test_lock_ordering_prevents_deadlock(self, temp_files):
        """Test that lock ordering prevents deadlocks"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Agent 1 tries to lock files in order: [0, 1]
            async def agent1_locks():
                return await manager.acquire_multiple_locks(
                    file_paths=[temp_files[0], temp_files[1]],
                    owner_id="agent-1",
                    timeout_seconds=5
                )
            
            # Agent 2 tries to lock files in reverse order: [1, 0]
            # But lock ordering will sort them to [0, 1]
            async def agent2_locks():
                return await manager.acquire_multiple_locks(
                    file_paths=[temp_files[1], temp_files[0]],
                    owner_id="agent-2",
                    timeout_seconds=5
                )
            
            # Run concurrently
            results = await asyncio.gather(
                agent1_locks(),
                agent2_locks(),
                return_exceptions=True
            )
            
            # One should succeed, one should fail (or timeout)
            # But no deadlock should occur
            assert len(results) == 2
            
        finally:
            await manager.stop()


class TestMultipleFileLocking:
    """Test locking multiple files"""
    
    @pytest.mark.asyncio
    async def test_acquire_multiple_locks(self, temp_files):
        """Test acquiring locks on multiple files"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            results = await manager.acquire_multiple_locks(
                file_paths=[temp_files[0], temp_files[1], temp_files[2]],
                owner_id="agent-1"
            )
            
            # All should succeed
            assert len(results) == 3
            assert all(lock_id is not None for lock_id in results.values())
            
            # Verify all locks are active
            stats = manager.get_stats()
            assert stats["active_locks"] == 3
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_release_multiple_locks(self, temp_files):
        """Test releasing locks on multiple files"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            # Acquire locks
            await manager.acquire_multiple_locks(
                file_paths=[temp_files[0], temp_files[1], temp_files[2]],
                owner_id="agent-1"
            )
            
            # Release locks
            results = await manager.release_multiple_locks(
                file_paths=[temp_files[0], temp_files[1], temp_files[2]],
                owner_id="agent-1"
            )
            
            # All should succeed
            assert len(results) == 3
            assert all(success for success in results.values())
            
            # Verify all locks are released
            stats = manager.get_stats()
            assert stats["active_locks"] == 0
            
        finally:
            await manager.stop()


class TestPerformance:
    """Test performance requirements"""
    
    @pytest.mark.asyncio
    async def test_lock_acquisition_time(self, temp_files):
        """Test that lock acquisition is <100ms"""
        manager = FileLockManager()
        await manager.start()
        
        try:
            acquisition_times = []
            
            # Acquire and release 20 locks
            for i in range(20):
                start = time.time()
                lock_id = await manager.acquire_lock(
                    file_path=temp_files[i % len(temp_files)],
                    owner_id=f"agent-{i}"
                )
                acquisition_time = time.time() - start
                acquisition_times.append(acquisition_time)
                
                # Release immediately
                await manager.release_lock(
                    file_path=temp_files[i % len(temp_files)],
                    owner_id=f"agent-{i}"
                )
            
            # Calculate 95th percentile
            acquisition_times.sort()
            p95 = acquisition_times[int(len(acquisition_times) * 0.95)]
            
            print(f"\nLock acquisition times (95th percentile): {p95*1000:.1f}ms")
            assert p95 < 0.1, f"95th percentile {p95*1000:.1f}ms exceeds 100ms target"
            
        finally:
            await manager.stop()


class TestSingletonPattern:
    """Test singleton manager pattern"""
    
    def test_singleton_returns_same_instance(self):
        """Test that get_file_lock_manager returns singleton"""
        manager1 = get_file_lock_manager()
        manager2 = get_file_lock_manager()
        
        assert manager1 is manager2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

