"""
Inter-Component Communication System for System Orchestration

This module provides inter-component communication functionality for the trading system,
enabling components to communicate with each other through various messaging patterns.
"""

import json
import logging
import threading
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import uuid
import pickle
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message type enumeration"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class MessagePriority(Enum):
    """Message priority enumeration"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class Message:
    """Message structure"""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: Optional[str]
    topic: Optional[str]
    priority: MessagePriority
    timestamp: datetime
    payload: Any
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds

@dataclass
class Subscription:
    """Message subscription"""
    subscriber_id: str
    topic: str
    callback: Callable
    message_types: List[MessageType]
    priority_filter: Optional[MessagePriority] = None
    created_at: datetime = None

class CommunicationManager:
    """
    Inter-component communication manager for handling message routing and delivery.
    """
    
    def __init__(self, manager_name: str = "trading_system_comm"):
        self.manager_name = manager_name
        self.components: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, List[Subscription]] = {}
        self.message_queues: Dict[str, queue.PriorityQueue] = {}
        self.routing_table: Dict[str, str] = {}
        
        # Message processing
        self.message_processor = None
        self.processing_threads = []
        self.max_workers = 10
        
        # Statistics
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'delivered': 0,
            'dropped': 0,
            'errors': 0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Message history (for debugging)
        self.message_history: List[Message] = []
        self.max_history = 1000
        
        logger.info(f"Communication manager '{manager_name}' initialized")
    
    def start(self):
        """Start the communication manager"""
        with self._lock:
            if not self._running:
                self._running = True
                self._start_message_processor()
                logger.info("Communication manager started")
    
    def stop(self):
        """Stop the communication manager"""
        with self._lock:
            if self._running:
                self._running = False
                self._stop_message_processor()
                self._executor.shutdown(wait=True)
                logger.info("Communication manager stopped")
    
    def register_component(self, component_id: str, component_name: str, 
                          message_handlers: Optional[Dict[str, Callable]] = None) -> bool:
        """
        Register a component for communication.
        
        Args:
            component_id: Unique component identifier
            component_name: Human-readable component name
            message_handlers: Optional message handlers
            
        Returns:
            True if component was registered successfully
        """
        with self._lock:
            if component_id in self.components:
                logger.warning(f"Component {component_id} already registered")
                return False
            
            # Register component
            self.components[component_id] = {
                'name': component_name,
                'registered_at': datetime.utcnow(),
                'message_handlers': message_handlers or {},
                'last_activity': datetime.utcnow()
            }
            
            # Create message queue
            self.message_queues[component_id] = queue.PriorityQueue()
            
            # Update routing table
            self.routing_table[component_name] = component_id
            
            logger.info(f"Component registered: {component_name} ({component_id})")
            return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from communication.
        
        Args:
            component_id: Component identifier to unregister
            
        Returns:
            True if component was unregistered successfully
        """
        with self._lock:
            if component_id not in self.components:
                return False
            
            component_name = self.components[component_id]['name']
            
            # Remove component
            del self.components[component_id]
            
            # Remove message queue
            if component_id in self.message_queues:
                del self.message_queues[component_id]
            
            # Remove from routing table
            if component_name in self.routing_table:
                del self.routing_table[component_name]
            
            # Remove subscriptions
            if component_id in self.subscriptions:
                del self.subscriptions[component_id]
            
            logger.info(f"Component unregistered: {component_name} ({component_id})")
            return True
    
    def send_message(self, sender: str, recipient: str, message_type: MessageType,
                    payload: Any, priority: MessagePriority = MessagePriority.MEDIUM,
                    topic: Optional[str] = None, correlation_id: Optional[str] = None,
                    reply_to: Optional[str] = None, ttl: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message to a specific component.
        
        Args:
            sender: Sender component ID
            recipient: Recipient component ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            topic: Optional topic
            correlation_id: Optional correlation ID
            reply_to: Optional reply-to address
            ttl: Optional time to live
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        
        message = Message(
            message_id=message_id,
            message_type=message_type,
            sender=sender,
            recipient=recipient,
            topic=topic,
            priority=priority,
            timestamp=datetime.utcnow(),
            payload=payload,
            metadata=metadata or {},
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl=ttl
        )
        
        # Add to history
        self._add_to_history(message)
        
        # Route message
        success = self._route_message(message)
        
        if success:
            self.message_stats['sent'] += 1
            logger.debug(f"Message sent: {message_id} from {sender} to {recipient}")
        else:
            self.message_stats['errors'] += 1
            logger.error(f"Failed to send message: {message_id}")
        
        return message_id
    
    def publish_event(self, sender: str, topic: str, payload: Any,
                     priority: MessagePriority = MessagePriority.MEDIUM,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Publish an event to all subscribers of a topic.
        
        Args:
            sender: Sender component ID
            topic: Event topic
            payload: Event payload
            priority: Message priority
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        
        message = Message(
            message_id=message_id,
            message_type=MessageType.EVENT,
            sender=sender,
            recipient=None,
            topic=topic,
            priority=priority,
            timestamp=datetime.utcnow(),
            payload=payload,
            metadata=metadata or {},
            correlation_id=None,
            reply_to=None,
            ttl=None
        )
        
        # Add to history
        self._add_to_history(message)
        
        # Publish to subscribers
        subscribers = self._get_subscribers(topic, MessageType.EVENT)
        delivered_count = 0
        
        for subscriber in subscribers:
            if self._deliver_message(subscriber.subscriber_id, message):
                delivered_count += 1
        
        self.message_stats['sent'] += 1
        self.message_stats['delivered'] += delivered_count
        
        logger.debug(f"Event published: {message_id} on topic {topic} to {delivered_count} subscribers")
        return message_id
    
    def subscribe(self, subscriber_id: str, topic: str, callback: Callable,
                 message_types: Optional[List[MessageType]] = None,
                 priority_filter: Optional[MessagePriority] = None) -> bool:
        """
        Subscribe to messages on a topic.
        
        Args:
            subscriber_id: Subscriber component ID
            topic: Topic to subscribe to
            callback: Callback function
            message_types: Optional message type filter
            priority_filter: Optional priority filter
            
        Returns:
            True if subscription was successful
        """
        with self._lock:
            if subscriber_id not in self.components:
                logger.error(f"Component {subscriber_id} not registered")
                return False
            
            if subscriber_id not in self.subscriptions:
                self.subscriptions[subscriber_id] = []
            
            subscription = Subscription(
                subscriber_id=subscriber_id,
                topic=topic,
                callback=callback,
                message_types=message_types or [MessageType.EVENT],
                priority_filter=priority_filter,
                created_at=datetime.utcnow()
            )
            
            self.subscriptions[subscriber_id].append(subscription)
            
            logger.info(f"Subscription created: {subscriber_id} -> {topic}")
            return True
    
    def unsubscribe(self, subscriber_id: str, topic: str) -> bool:
        """
        Unsubscribe from messages on a topic.
        
        Args:
            subscriber_id: Subscriber component ID
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscription was successful
        """
        with self._lock:
            if subscriber_id not in self.subscriptions:
                return False
            
            # Remove matching subscriptions
            original_count = len(self.subscriptions[subscriber_id])
            self.subscriptions[subscriber_id] = [
                sub for sub in self.subscriptions[subscriber_id]
                if not (sub.topic == topic and sub.subscriber_id == subscriber_id)
            ]
            
            removed_count = original_count - len(self.subscriptions[subscriber_id])
            
            if removed_count > 0:
                logger.info(f"Unsubscribed: {subscriber_id} from {topic} ({removed_count} subscriptions)")
                return True
            
            return False
    
    def send_request(self, sender: str, recipient: str, payload: Any,
                    timeout: float = 30.0, priority: MessagePriority = MessagePriority.MEDIUM,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Send a request and wait for response.
        
        Args:
            sender: Sender component ID
            recipient: Recipient component ID
            payload: Request payload
            timeout: Response timeout in seconds
            priority: Message priority
            metadata: Optional metadata
            
        Returns:
            Response payload or None if timeout
        """
        correlation_id = str(uuid.uuid4())
        reply_queue = queue.Queue()
        
        # Store reply queue
        with self._lock:
            if not hasattr(self, '_reply_queues'):
                self._reply_queues = {}
            self._reply_queues[correlation_id] = reply_queue
        
        try:
            # Send request
            message_id = self.send_message(
                sender=sender,
                recipient=recipient,
                message_type=MessageType.REQUEST,
                payload=payload,
                priority=priority,
                correlation_id=correlation_id,
                reply_to=sender,
                metadata=metadata
            )
            
            # Wait for response
            try:
                response = reply_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                logger.warning(f"Request timeout: {message_id}")
                return None
                
        finally:
            # Clean up reply queue
            with self._lock:
                if hasattr(self, '_reply_queues') and correlation_id in self._reply_queues:
                    del self._reply_queues[correlation_id]
    
    def send_response(self, sender: str, recipient: str, payload: Any,
                     correlation_id: str, priority: MessagePriority = MessagePriority.MEDIUM,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a response to a request.
        
        Args:
            sender: Sender component ID
            recipient: Recipient component ID
            payload: Response payload
            correlation_id: Correlation ID from original request
            priority: Message priority
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        return self.send_message(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.RESPONSE,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            metadata=metadata
        )
    
    def get_message(self, component_id: str, timeout: float = 1.0) -> Optional[Message]:
        """
        Get the next message for a component.
        
        Args:
            component_id: Component ID
            timeout: Timeout in seconds
            
        Returns:
            Message or None if timeout
        """
        if component_id not in self.message_queues:
            return None
        
        try:
            # Get message from queue (priority, timestamp, message)
            priority, timestamp, message = self.message_queues[component_id].get(timeout=timeout)
            
            # Check TTL
            if message.ttl and (datetime.utcnow() - message.timestamp).total_seconds() > message.ttl:
                logger.debug(f"Message expired: {message.message_id}")
                self.message_stats['dropped'] += 1
                return self.get_message(component_id, timeout)
            
            self.message_stats['received'] += 1
            return message
            
        except queue.Empty:
            return None
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.
        
        Returns:
            Communication statistics
        """
        with self._lock:
            total_components = len(self.components)
            total_subscriptions = sum(len(subs) for subs in self.subscriptions.values())
            
            return {
                'total_components': total_components,
                'total_subscriptions': total_subscriptions,
                'message_stats': self.message_stats.copy(),
                'message_history_size': len(self.message_history),
                'active_queues': len(self.message_queues)
            }
    
    def _start_message_processor(self):
        """Start message processing threads"""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._message_processor_worker, daemon=True)
            thread.start()
            self.processing_threads.append(thread)
    
    def _stop_message_processor(self):
        """Stop message processing threads"""
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        self.processing_threads.clear()
    
    def _message_processor_worker(self):
        """Message processing worker thread"""
        while self._running:
            try:
                # Process messages for all components
                for component_id in list(self.components.keys()):
                    if component_id in self.message_queues:
                        message = self.get_message(component_id, timeout=0.1)
                        if message:
                            self._process_message(component_id, message)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in message processor worker: {e}")
    
    def _process_message(self, component_id: str, message: Message):
        """Process a message for a component"""
        try:
            component = self.components[component_id]
            component['last_activity'] = datetime.utcnow()
            
            # Handle response messages
            if message.message_type == MessageType.RESPONSE:
                self._handle_response(message)
                return
            
            # Call message handlers
            handlers = component.get('message_handlers', {})
            
            # Try specific handler for message type
            handler_key = f"handle_{message.message_type.value}"
            if handler_key in handlers:
                handlers[handler_key](message)
                return
            
            # Try general message handler
            if 'handle_message' in handlers:
                handlers['handle_message'](message)
                return
            
            # Default handling
            logger.debug(f"No handler for message type {message.message_type.value} in component {component_id}")
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id} for component {component_id}: {e}")
            self.message_stats['errors'] += 1
    
    def _handle_response(self, message: Message):
        """Handle response messages"""
        if not hasattr(self, '_reply_queues'):
            return
        
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self._reply_queues:
            try:
                self._reply_queues[correlation_id].put(message.payload)
            except Exception as e:
                logger.error(f"Error handling response: {e}")
    
    def _route_message(self, message: Message) -> bool:
        """Route a message to its destination"""
        try:
            if message.recipient:
                # Direct message
                return self._deliver_message(message.recipient, message)
            elif message.topic:
                # Topic-based message
                subscribers = self._get_subscribers(message.topic, message.message_type)
                success = False
                for subscriber in subscribers:
                    if self._deliver_message(subscriber.subscriber_id, message):
                        success = True
                return success
            else:
                logger.error(f"Message has no recipient or topic: {message.message_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error routing message {message.message_id}: {e}")
            return False
    
    def _deliver_message(self, component_id: str, message: Message) -> bool:
        """Deliver a message to a component"""
        if component_id not in self.message_queues:
            return False
        
        try:
            # Priority queue: (priority, timestamp, message)
            priority = message.priority.value
            timestamp = message.timestamp.timestamp()
            
            self.message_queues[component_id].put((priority, timestamp, message))
            return True
            
        except Exception as e:
            logger.error(f"Error delivering message {message.message_id} to {component_id}: {e}")
            return False
    
    def _get_subscribers(self, topic: str, message_type: MessageType) -> List[Subscription]:
        """Get subscribers for a topic and message type"""
        subscribers = []
        
        for component_subscriptions in self.subscriptions.values():
            for subscription in component_subscriptions:
                if (subscription.topic == topic and 
                    message_type in subscription.message_types):
                    subscribers.append(subscription)
        
        return subscribers
    
    def _add_to_history(self, message: Message):
        """Add message to history"""
        self.message_history.append(message)
        
        # Trim history if too long
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]


# Global instance management
_communication_manager_instance = None

def get_communication_manager() -> CommunicationManager:
    """Get or create communication manager instance"""
    global _communication_manager_instance
    
    if _communication_manager_instance is None:
        _communication_manager_instance = CommunicationManager()
        _communication_manager_instance.start()
    
    return _communication_manager_instance


def init_communication_manager(manager_name: str = "trading_system_comm") -> CommunicationManager:
    """Initialize communication manager with custom name"""
    global _communication_manager_instance
    
    if _communication_manager_instance:
        _communication_manager_instance.stop()
    
    _communication_manager_instance = CommunicationManager(manager_name)
    _communication_manager_instance.start()
    
    return _communication_manager_instance