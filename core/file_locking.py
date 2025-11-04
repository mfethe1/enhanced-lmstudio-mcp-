"""
File-Level Locking System

Implements file-level locking to prevent concurrent modifications by multiple agents:
- Lock acquisition with timeout (default 60s)
- Lock release (automatic and manual)
- Lock status monitoring and conflict detection
- Deadlock prevention via lock ordering
- Integration with plan execution
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LockState(Enum):
    """File lock states"""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    PENDING = "pending"          # Waiting for lock
    EXPIRED = "expired"          # Timeout occurred
    FORCE_RELEASED = "force_released"  # Manually released


@dataclass
class FileLock:
    """Represents a lock on a file"""
    file_path: str
    state: LockState
    owner_id: str  # Agent ID or task ID
    acquired_at: float
    expires_at: float
    timeout_seconds: int = 60
    lock_id: str = field(default_factory=lambda: f"lock-{uuid.uuid4().hex[:8]}")
    retry_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if lock has expired"""
        return time.time() > self.expires_at
    
    def time_remaining(self) -> float:
        """Get remaining time before expiration"""
        return max(0.0, self.expires_at - time.time())
    
    def hold_duration(self) -> float:
        """Get how long lock has been held"""
        return time.time() - self.acquired_at


@dataclass
class LockRequest:
    """Represents a request for a file lock"""
    request_id: str
    file_path: str
    owner_id: str
    priority: int = 0  # Higher = more urgent
    created_at: float = field(default_factory=time.time)
    timeout_seconds: int = 60
    callback: Optional[Callable] = None
    
    def is_expired(self) -> bool:
        """Check if request has timed out"""
        return (time.time() - self.created_at) > self.timeout_seconds
    
    def __lt__(self, other):
        """For priority queue ordering (higher priority first)"""
        return self.priority > other.priority


class FileLockManager:
    """
    Manages file-level locks with timeout and deadlock prevention.
    
    Features:
    - Lock acquisition with timeout (default 60s)
    - Automatic lock release on timeout
    - Deadlock prevention via lock ordering
    - Conflict detection and monitoring
    - Performance: lock acquisition <100ms
    """
    
    def __init__(self, default_timeout: int = 60):
        self.default_timeout = default_timeout
        self.locks: Dict[str, FileLock] = {}  # file_path -> FileLock
        self.pending_requests: Dict[str, List[LockRequest]] = {}  # file_path -> [requests]
        self.lock = asyncio.Lock()  # Protect registry modifications
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_force_released = 0
        self.total_conflicts = 0
        self.acquisition_times: List[float] = []
    
    async def start(self):
        """Start background cleanup task"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"FileLockManager started (default_timeout={self.default_timeout}s)")
    
    async def stop(self):
        """Stop background tasks and release all locks"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Release all locks
        async with self.lock:
            for file_path in list(self.locks.keys()):
                await self._release_lock_internal(file_path, force=True, reason="Manager stopped")
        
        logger.info("FileLockManager stopped")
    
    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path for consistent comparison"""
        return str(Path(file_path).resolve())
    
    async def acquire_lock(
        self,
        file_path: str,
        owner_id: str,
        timeout_seconds: Optional[int] = None,
        priority: int = 0,
        wait: bool = True
    ) -> Optional[str]:
        """
        Acquire a lock on a file.
        
        Args:
            file_path: Path to file to lock
            owner_id: ID of agent/task requesting lock
            timeout_seconds: Lock timeout (default: manager default)
            priority: Request priority (higher = more urgent)
            wait: If True, wait for lock; if False, fail immediately if locked
            
        Returns:
            lock_id if successful, None if failed
            
        Raises:
            TimeoutError: If wait timeout exceeded
            FileNotFoundError: If file doesn't exist
        """
        file_path = self._normalize_path(file_path)
        timeout_seconds = timeout_seconds or self.default_timeout
        start_time = time.time()
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        async with self.lock:
            # Check if already locked
            if file_path in self.locks:
                existing_lock = self.locks[file_path]
                
                # Check if same owner (re-entrant lock)
                if existing_lock.owner_id == owner_id:
                    logger.info(f"Re-entrant lock for {file_path} by {owner_id}")
                    return existing_lock.lock_id
                
                # File is locked by someone else
                if not wait:
                    logger.warning(f"Lock conflict: {file_path} locked by {existing_lock.owner_id}")
                    self.total_conflicts += 1
                    return None
                
                # Add to pending requests
                request = LockRequest(
                    request_id=f"req-{uuid.uuid4().hex[:8]}",
                    file_path=file_path,
                    owner_id=owner_id,
                    priority=priority,
                    timeout_seconds=timeout_seconds
                )
                
                if file_path not in self.pending_requests:
                    self.pending_requests[file_path] = []
                self.pending_requests[file_path].append(request)
                self.pending_requests[file_path].sort(key=lambda r: r.priority, reverse=True)
                
                logger.info(f"Lock request queued: {file_path} by {owner_id} (priority={priority})")
                self.total_conflicts += 1
            
            else:
                # File is unlocked, acquire immediately
                lock = FileLock(
                    file_path=file_path,
                    state=LockState.LOCKED,
                    owner_id=owner_id,
                    acquired_at=time.time(),
                    expires_at=time.time() + timeout_seconds,
                    timeout_seconds=timeout_seconds
                )
                self.locks[file_path] = lock
                
                acquisition_time = time.time() - start_time
                self.acquisition_times.append(acquisition_time)
                self.total_acquired += 1
                
                logger.info(f"Lock acquired: {file_path} by {owner_id} (timeout={timeout_seconds}s, time={acquisition_time*1000:.1f}ms)")
                return lock.lock_id
        
        # Wait for lock if requested
        if wait:
            return await self._wait_for_lock(file_path, owner_id, timeout_seconds)
        
        return None
    
    async def _wait_for_lock(
        self,
        file_path: str,
        owner_id: str,
        timeout_seconds: int
    ) -> Optional[str]:
        """Wait for a lock to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            await asyncio.sleep(0.1)  # Check every 100ms
            
            async with self.lock:
                # Check if lock is now available
                if file_path not in self.locks:
                    # Check if we're next in queue
                    if file_path in self.pending_requests and self.pending_requests[file_path]:
                        next_request = self.pending_requests[file_path][0]
                        if next_request.owner_id == owner_id:
                            # We're next! Acquire lock
                            self.pending_requests[file_path].pop(0)
                            if not self.pending_requests[file_path]:
                                del self.pending_requests[file_path]
                            
                            lock = FileLock(
                                file_path=file_path,
                                state=LockState.LOCKED,
                                owner_id=owner_id,
                                acquired_at=time.time(),
                                expires_at=time.time() + timeout_seconds,
                                timeout_seconds=timeout_seconds
                            )
                            self.locks[file_path] = lock
                            self.total_acquired += 1
                            
                            logger.info(f"Lock acquired from queue: {file_path} by {owner_id}")
                            return lock.lock_id
        
        # Timeout occurred
        logger.warning(f"Lock acquisition timeout: {file_path} by {owner_id}")
        self.total_timeouts += 1
        
        # Remove from pending requests
        async with self.lock:
            if file_path in self.pending_requests:
                self.pending_requests[file_path] = [
                    r for r in self.pending_requests[file_path]
                    if r.owner_id != owner_id
                ]
                if not self.pending_requests[file_path]:
                    del self.pending_requests[file_path]
        
        raise TimeoutError(f"Lock acquisition timeout for {file_path} after {timeout_seconds}s")

    async def release_lock(
        self,
        file_path: str,
        owner_id: str,
        force: bool = False
    ) -> bool:
        """
        Release a lock on a file.

        Args:
            file_path: Path to file to unlock
            owner_id: ID of agent/task releasing lock
            force: If True, release even if owner doesn't match

        Returns:
            True if released, False if not found or owner mismatch
        """
        file_path = self._normalize_path(file_path)

        async with self.lock:
            return await self._release_lock_internal(file_path, owner_id, force)

    async def _release_lock_internal(
        self,
        file_path: str,
        owner_id: Optional[str] = None,
        force: bool = False,
        reason: str = "Normal release"
    ) -> bool:
        """Internal lock release (assumes lock held)"""
        if file_path not in self.locks:
            logger.warning(f"Lock not found for release: {file_path}")
            return False

        lock = self.locks[file_path]

        # Verify owner unless force
        if not force and owner_id and lock.owner_id != owner_id:
            logger.warning(f"Lock owner mismatch: {file_path} owned by {lock.owner_id}, release requested by {owner_id}")
            return False

        # Release lock
        hold_duration = lock.hold_duration()
        del self.locks[file_path]
        self.total_released += 1

        if force:
            self.total_force_released += 1

        logger.info(f"Lock released: {file_path} by {lock.owner_id} (duration={hold_duration:.1f}s, reason={reason})")

        # Process next pending request if any
        if file_path in self.pending_requests and self.pending_requests[file_path]:
            # Next request will be processed in _wait_for_lock
            pass

        return True

    async def acquire_multiple_locks(
        self,
        file_paths: List[str],
        owner_id: str,
        timeout_seconds: Optional[int] = None
    ) -> Dict[str, Optional[str]]:
        """
        Acquire locks on multiple files with deadlock prevention.

        Uses lock ordering (alphabetical) to prevent deadlocks.

        Args:
            file_paths: List of file paths to lock
            owner_id: ID of agent/task requesting locks
            timeout_seconds: Lock timeout for each file

        Returns:
            Dict mapping file_path to lock_id (None if failed)
        """
        # Sort paths alphabetically for lock ordering (deadlock prevention)
        sorted_paths = sorted([self._normalize_path(p) for p in file_paths])

        results = {}
        acquired_locks = []

        try:
            for file_path in sorted_paths:
                try:
                    lock_id = await self.acquire_lock(
                        file_path=file_path,
                        owner_id=owner_id,
                        timeout_seconds=timeout_seconds,
                        wait=True
                    )
                    results[file_path] = lock_id
                    if lock_id:
                        acquired_locks.append(file_path)
                except Exception as e:
                    logger.error(f"Failed to acquire lock on {file_path}: {e}")
                    results[file_path] = None
                    # Rollback: release all acquired locks
                    for acquired_path in acquired_locks:
                        await self.release_lock(acquired_path, owner_id, force=True)
                    raise

            return results

        except Exception as e:
            logger.error(f"Failed to acquire multiple locks: {e}")
            raise

    async def release_multiple_locks(
        self,
        file_paths: List[str],
        owner_id: str
    ) -> Dict[str, bool]:
        """
        Release locks on multiple files.

        Args:
            file_paths: List of file paths to unlock
            owner_id: ID of agent/task releasing locks

        Returns:
            Dict mapping file_path to success status
        """
        results = {}

        for file_path in file_paths:
            try:
                success = await self.release_lock(file_path, owner_id)
                results[file_path] = success
            except Exception as e:
                logger.error(f"Failed to release lock on {file_path}: {e}")
                results[file_path] = False

        return results

    async def _cleanup_loop(self):
        """Background task to cleanup expired locks"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                async with self.lock:
                    expired = [
                        file_path for file_path, lock in self.locks.items()
                        if lock.is_expired()
                    ]

                    for file_path in expired:
                        logger.warning(f"Lock expired: {file_path}")
                        await self._release_lock_internal(
                            file_path,
                            force=True,
                            reason="Timeout expired"
                        )
                        self.total_timeouts += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        avg_acquisition_time = (
            sum(self.acquisition_times[-100:]) / len(self.acquisition_times[-100:])
            if self.acquisition_times else 0.0
        )

        return {
            "active_locks": len(self.locks),
            "pending_requests": sum(len(reqs) for reqs in self.pending_requests.values()),
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "total_timeouts": self.total_timeouts,
            "total_force_released": self.total_force_released,
            "total_conflicts": self.total_conflicts,
            "avg_acquisition_time_ms": avg_acquisition_time * 1000,
            "locks": [
                {
                    "file_path": lock.file_path,
                    "owner_id": lock.owner_id,
                    "state": lock.state.value,
                    "hold_duration_seconds": lock.hold_duration(),
                    "time_remaining_seconds": lock.time_remaining(),
                    "lock_id": lock.lock_id
                }
                for lock in self.locks.values()
            ]
        }

    def get_lock_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific lock"""
        file_path = self._normalize_path(file_path)

        if file_path not in self.locks:
            return None

        lock = self.locks[file_path]
        return {
            "file_path": lock.file_path,
            "owner_id": lock.owner_id,
            "state": lock.state.value,
            "acquired_at": lock.acquired_at,
            "expires_at": lock.expires_at,
            "hold_duration_seconds": lock.hold_duration(),
            "time_remaining_seconds": lock.time_remaining(),
            "lock_id": lock.lock_id
        }


# Global singleton instance
_manager_singleton: Optional[FileLockManager] = None


def get_file_lock_manager(default_timeout: int = 60) -> FileLockManager:
    """Get or create the global file lock manager singleton"""
    global _manager_singleton

    if _manager_singleton is None:
        _manager_singleton = FileLockManager(default_timeout=default_timeout)

    return _manager_singleton

