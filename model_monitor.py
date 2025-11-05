"""
LM Studio model availability monitoring and alerting.
Tracks model changes and provides notifications when models become unavailable.
"""

import time
import threading
import logging
from typing import Set, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelChangeEvent:
    timestamp: float
    event_type: str  # 'added', 'removed', 'changed'
    model_name: str
    previous_models: Set[str] = field(default_factory=set)
    current_models: Set[str] = field(default_factory=set)


class ModelMonitor:
    """Monitors LM Studio model availability and triggers alerts on changes"""
    
    def __init__(self, server, check_interval: float = 300.0):  # 5 minutes default
        self.server = server
        self.check_interval = check_interval
        
        self._current_models: Set[str] = set()
        self._last_check: float = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, Callable] = {}
        self._events: list[ModelChangeEvent] = []
        self._lock = threading.Lock()
        
        # Initialize with current models
        self._update_models()
        
        logger.info(f"Model monitor initialized with {len(self._current_models)} models")
    
    def start_monitoring(self):
        """Start background model monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop background model monitoring"""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Model monitoring stopped")
    
    def add_callback(self, name: str, callback: Callable[[ModelChangeEvent], None]):
        """Add a callback function to be called on model changes"""
        with self._lock:
            self._callbacks[name] = callback
        logger.debug(f"Added model change callback: {name}")
    
    def remove_callback(self, name: str):
        """Remove a callback function"""
        with self._lock:
            self._callbacks.pop(name, None)
        logger.debug(f"Removed model change callback: {name}")
    
    def get_current_models(self) -> Set[str]:
        """Get the current set of available models"""
        with self._lock:
            return self._current_models.copy()
    
    def get_recent_events(self, limit: int = 10) -> list[ModelChangeEvent]:
        """Get recent model change events"""
        with self._lock:
            return self._events[-limit:] if self._events else []
    
    def force_check(self) -> bool:
        """Force an immediate model availability check"""
        return self._update_models()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                self._update_models()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in model monitoring loop: {e}")
                time.sleep(min(self.check_interval, 60.0))  # Don't spam on errors
    
    def _update_models(self) -> bool:
        """Update the current model list and detect changes"""
        try:
            # Get current models from LM Studio
            new_models = set(self.server.refresh_lmstudio_models(force=True))
            
            with self._lock:
                previous_models = self._current_models.copy()
                
                # Detect changes
                added_models = new_models - previous_models
                removed_models = previous_models - new_models
                
                # Update current models
                self._current_models = new_models
                self._last_check = time.time()
                
                # Create events for changes
                events_created = []
                
                for model in added_models:
                    event = ModelChangeEvent(
                        timestamp=self._last_check,
                        event_type='added',
                        model_name=model,
                        previous_models=previous_models,
                        current_models=new_models
                    )
                    self._events.append(event)
                    events_created.append(event)
                    logger.info(f"Model added: {model}")
                
                for model in removed_models:
                    event = ModelChangeEvent(
                        timestamp=self._last_check,
                        event_type='removed',
                        model_name=model,
                        previous_models=previous_models,
                        current_models=new_models
                    )
                    self._events.append(event)
                    events_created.append(event)
                    logger.warning(f"Model removed: {model}")
                
                # Trim event history
                if len(self._events) > 100:
                    self._events = self._events[-50:]
                
                # Trigger callbacks for each event
                for event in events_created:
                    for callback_name, callback in self._callbacks.items():
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Error in model change callback '{callback_name}': {e}")
                
                return len(events_created) > 0
        
        except Exception as e:
            logger.error(f"Error updating model list: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get monitoring statistics"""
        with self._lock:
            return {
                "monitoring": self._monitoring,
                "current_models": list(self._current_models),
                "model_count": len(self._current_models),
                "last_check": self._last_check,
                "last_check_time": datetime.fromtimestamp(self._last_check).isoformat() if self._last_check else None,
                "recent_events": len(self._events),
                "callbacks": list(self._callbacks.keys())
            }


def create_alert_callback(alert_threshold: int = 1) -> Callable:
    """Create a callback that logs alerts when models are removed"""
    def alert_callback(event: ModelChangeEvent):
        if event.event_type == 'removed':
            remaining_count = len(event.current_models)
            if remaining_count <= alert_threshold:
                logger.critical(f"ALERT: Model '{event.model_name}' removed. Only {remaining_count} models remaining!")
            else:
                logger.warning(f"Model '{event.model_name}' removed. {remaining_count} models remaining.")
        elif event.event_type == 'added':
            logger.info(f"Model '{event.model_name}' added. {len(event.current_models)} models available.")
    
    return alert_callback


def create_configured_model_callback(configured_models: Set[str]) -> Callable:
    """Create a callback that alerts when configured models become unavailable"""
    def configured_callback(event: ModelChangeEvent):
        if event.event_type == 'removed' and event.model_name in configured_models:
            logger.critical(f"CRITICAL: Configured model '{event.model_name}' is no longer available!")
            logger.critical(f"Available models: {', '.join(sorted(event.current_models))}")
    
    return configured_callback
