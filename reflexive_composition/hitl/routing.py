# reflexive_composition/hitl/routing.py
"""
Validation routing and task assignment.

This module handles the routing of validation tasks to appropriate
validators based on task type, domain, and other factors.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable

logger = logging.getLogger(__name__)

class ValidationRouter:
    """
    Routes validation tasks to appropriate validators.
    
    This class handles the assignment of validation tasks to specific
    validators based on task type, domain expertise, and availability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation router.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize task queues
        self.task_queues = {
            "general": queue.Queue(),
            "schema": queue.Queue(),
            "contradiction": queue.Queue(),
            "response": queue.Queue()
        }
        
        # Initialize validator registry
        self.validators = {}
        
        # Configure async mode
        self.async_mode = self.config.get("async_validation", False)
        if self.async_mode:
            self.results = {}  # task_id -> result
            self.result_lock = threading.Lock()
            
            # Start worker threads
            self.workers = []
            num_workers = self.config.get("num_workers", 1)
            for i in range(num_workers):
                worker = threading.Thread(target=self._validation_worker, daemon=True)
                worker.start()
                self.workers.append(worker)
    
    def route_validation(self,
                       validator_type: str, 
                       context: Dict[str, Any],
                       interface: Any,
                       interactive: bool = True) -> Dict[str, Any]:
        """
        Route a validation task to the appropriate validator.
        
        Args:
            validator_type: Type of validator needed
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        # Check if we have a specific validator for this type
        if validator_type in self.validators:
            return self._present_to_validator(
                self.validators[validator_type], context, interface, interactive=interactive
            )
        
        # Otherwise, handle based on validator type
        if validator_type == "general":
            return self._handle_general_validation(context, interface, interactive=interactive)
        elif validator_type == "schema":
            return self._handle_schema_validation(context, interface)
        elif validator_type == "contradiction":
            return self._handle_contradiction_validation(context, interface, interactive=interactive)
        elif validator_type == "response":
            return self._handle_response_validation(context, interface)
        else:
            logger.warning(f"Unsupported validator type: {validator_type}")
            return {
                "accepted": False,
                "validation_type": "unsupported",
                "reason": f"Unsupported validator type: {validator_type}"
            }
    
    def register_validator(self, 
                         validator_type: str, 
                         validator_func: Callable[[Dict[str, Any], Any], Dict[str, Any]]) -> None:
        """
        Register a custom validator function.
        
        Args:
            validator_type: Type of validator
            validator_func: Validator function
        """
        self.validators[validator_type] = validator_func
        logger.info(f"Registered custom validator for type: {validator_type}")
    
    def queue_validation_task(self, 
                            task_id: str,
                            validator_type: str, 
                            context: Dict[str, Any],
                            interface: Any) -> str:
        """
        Queue a validation task for asynchronous processing.
        
        Args:
            task_id: Unique task identifier
            validator_type: Type of validator needed
            context: Validation context
            interface: Validation interface
            
        Returns:
            Task ID for result retrieval
        """
        if not self.async_mode:
            logger.warning("Queuing validation task in synchronous mode")
        
        # Create task object
        task = {
            "id": task_id,
            "type": validator_type,
            "context": context,
            "interface": interface,
            "timestamp": time.time()
        }
        
        # Add to appropriate queue
        if validator_type in self.task_queues:
            self.task_queues[validator_type].put(task)
        else:
            # Default to general queue
            self.task_queues["general"].put(task)
        
        return task_id
    
    def get_validation_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of an asynchronous validation task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Validation result or None if not ready
        """
        if not self.async_mode:
            logger.warning("Getting validation result in synchronous mode")
            return None
        
        start_time = time.time()
        
        while timeout is None or time.time() - start_time < timeout:
            with self.result_lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
        
        return None
    
    def _validation_worker(self) -> None:
        """
        Worker thread for processing validation tasks.
        """
        while True:
            # Check all queues in priority order
            for queue_type in ["contradiction", "schema", "response", "general"]:
                try:
                    # Non-blocking check
                    task = self.task_queues[queue_type].get_nowait()
                    
                    # Process the task
                    result = self.route_validation(
                        task["type"], task["context"], task["interface"]
                    )
                    
                    # Store the result
                    with self.result_lock:
                        self.results[task["id"]] = result
                    
                    # Mark task as done
                    self.task_queues[queue_type].task_done()
                    
                    # Break out of the loop and start again
                    break
                    
                except queue.Empty:
                    # Queue is empty, try the next one
                    continue
            
            # If we checked all queues and found nothing, sleep briefly
            time.sleep(0.1)
    
    def _present_to_validator(self,
                              validator_func: Callable, 
                              context: Dict[str, Any],
                              interface: Any,
                              interactive: bool = True) -> Dict[str, Any]:
        """
        Present a validation task to a validator function.
        
        Args:
            validator_func: Validator function
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        try:
            return validator_func(context, interface, interactive=interactive)
        except Exception as e:
            logger.error(f"Error in validator function: {e}")
            return {
                "accepted": False,
                "validation_type": "error",
                "reason": f"Validator error: {str(e)}"
            }
    
    def _handle_general_validation(self, 
                                 context: Dict[str, Any],
                                 interface: Any,
                                 interactive: bool = True) -> Dict[str, Any]:
        """
        Handle general validation tasks.
        
        Args:
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        # Check if we're validating a triple
        if "triple" in context:
            return interface.present_triple_validation(context, interactive=interactive)
        
        # Otherwise, default behavior
        logger.warning("General validation with unknown context structure")
        return {
            "accepted": False,
            "validation_type": "unknown_context",
            "reason": "Unknown context structure in general validation"
        }
    
    def _handle_schema_validation(self, 
                                context: Dict[str, Any],
                                interface: Any) -> Dict[str, Any]:
        """
        Handle schema validation tasks.
        
        Args:
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        return interface.present_schema_validation(context)
    
    def _handle_contradiction_validation(self, 
                                       context: Dict[str, Any],
                                       interface: Any,
                                       interactive: bool = True) -> Dict[str, Any]:
        """
        Handle contradiction validation tasks.
        
        Args:
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        # Add contradiction info to the context for better visibility
        context_with_contradiction = context.copy()
        
        if "contradiction" in context and context["contradiction"]:
            contradiction = context["contradiction"]
            context_with_contradiction["contradiction_details"] = contradiction
        
        return interface.present_triple_validation(context_with_contradiction, interactive=interactive)
    
    def _handle_response_validation(self, 
                                  context: Dict[str, Any],
                                  interface: Any) -> Dict[str, Any]:
        """
        Handle response validation tasks.
        
        Args:
            context: Validation context
            interface: Validation interface
            
        Returns:
            Validation result
        """
        return interface.present_response_validation(context)