"""
MFE Toolbox UI - Task Manager Module

This module provides advanced task management capabilities for the MFE Toolbox UI,
including task prioritization, scheduling, batching, and dependency management.
It offers a higher-level abstraction over the Worker and AsyncWorker classes to
orchestrate complex computational workflows while maintaining UI responsiveness.
"""

import asyncio
import enum
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
import heapq

from PyQt6.QtCore import QObject, QThreadPool, pyqtSignal, pyqtSlot

from .worker import Worker, AsyncWorker
from .signals import TaskSignals, SignalFactory

# Configure module logger
logger = logging.getLogger(__name__)


class TaskPriority(enum.Enum):
    """Enumeration of task priority levels for ordering task execution."""
    HIGH = 10
    NORMAL = 20
    LOW = 30
    BACKGROUND = 40


class TaskStatus(enum.Enum):
    """Enumeration of task status values for tracking task lifecycle state."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING_DEPENDENCY = "WAITING_DEPENDENCY"


class TaskType(enum.Enum):
    """Enumeration of task types for different categories of computational tasks."""
    GENERAL = "GENERAL"
    MODEL = "MODEL"
    PLOT = "PLOT"
    DATA_PROCESSING = "DATA_PROCESSING"


@dataclass
class Task:
    """Dataclass representing a task with its metadata, used for scheduling and tracking."""
    task_id: str
    name: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    task_type: TaskType
    dependencies: list
    is_async: bool

    # Runtime state
    status: TaskStatus = field(init=False)
    worker: Optional[Union[Worker, AsyncWorker]] = field(default=None, init=False)
    progress: float = field(default=0.0, init=False)
    result: Optional[Any] = field(default=None, init=False)
    error: Optional[Exception] = field(default=None, init=False)

    def __post_init__(self):
        """Set initial status based on dependencies."""
        if self.dependencies and len(self.dependencies) > 0:
            self.status = TaskStatus.WAITING_DEPENDENCY
        else:
            self.status = TaskStatus.PENDING


class TaskGroup:
    """Class for managing a group of related tasks that can be controlled together."""
    
    def __init__(self, group_id: str, name: str):
        """Initialize a task group."""
        self.group_id = group_id
        self.name = name
        self.task_ids = []
        self.is_cancelled = False
    
    def add_task(self, task_id: str) -> None:
        """Add a task to the group."""
        self.task_ids.append(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the group."""
        if task_id in self.task_ids:
            self.task_ids.remove(task_id)
            return True
        return False
    
    def get_progress(self, tasks_dict: dict) -> float:
        """Calculate the aggregate progress of all tasks in the group."""
        if not self.task_ids:
            return 0.0
        
        total_progress = 0.0
        valid_tasks = 0
        
        for task_id in self.task_ids:
            if task_id in tasks_dict:
                task = tasks_dict[task_id]
                
                if task.status == TaskStatus.COMPLETED:
                    total_progress += 1.0
                elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                    # Don't count failed/cancelled tasks in progress calculation
                    pass
                else:
                    total_progress += task.progress
                
                valid_tasks += 1
        
        if valid_tasks == 0:
            return 0.0
        
        return total_progress / valid_tasks


class TaskManagerSignals(QObject):
    """Signal container for TaskManager events and notifications."""
    
    # Task signals
    task_started = pyqtSignal(str)  # task_id
    task_completed = pyqtSignal(str, object)  # task_id, result
    task_failed = pyqtSignal(str, Exception)  # task_id, error
    task_cancelled = pyqtSignal(str)  # task_id
    task_progress = pyqtSignal(str, float)  # task_id, progress
    
    # Group signals
    group_started = pyqtSignal(str)  # group_id
    group_completed = pyqtSignal(str)  # group_id
    group_progress = pyqtSignal(str, float)  # group_id, progress
    group_cancelled = pyqtSignal(str)  # group_id
    
    # Management signals
    active_tasks_changed = pyqtSignal(int)  # active_count
    
    def __init__(self):
        """Initialize the task manager signals."""
        super().__init__()


class TaskManager(QObject):
    """
    Advanced task management system for scheduling, prioritizing, and executing tasks
    with dependency tracking and group operations.
    """
    
    def __init__(self, max_concurrent_tasks: int = None):
        """
        Initialize the task manager.
        
        Parameters
        ----------
        max_concurrent_tasks : int, optional
            Maximum number of tasks that can run concurrently. 
            If None, defaults to QThreadPool's maxThreadCount.
        """
        super().__init__()
        
        # Task tracking
        self._tasks = {}  # task_id -> Task
        self._groups = {}  # group_id -> TaskGroup
        self._task_queue = []  # Priority queue for pending tasks
        
        # Thread pool for task execution
        self._thread_pool = QThreadPool.globalInstance()
        self._max_concurrent_tasks = max_concurrent_tasks or self._thread_pool.maxThreadCount()
        self._active_task_count = 0
        
        # Signals for events
        self.signals = TaskManagerSignals()
        self._signal_factory = SignalFactory()
        
        logger.debug(f"TaskManager initialized with max_concurrent_tasks={self._max_concurrent_tasks}")
    
    def create_task(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_type: TaskType = TaskType.GENERAL,
        dependencies: list = None,
        group_id: str = None
    ) -> str:
        """
        Create a new task and add it to the task manager.
        
        Parameters
        ----------
        name : str
            Name of the task for display and tracking
        function : Callable
            The function to execute
        args : tuple, optional
            Positional arguments for the function
        kwargs : dict, optional
            Keyword arguments for the function
        priority : TaskPriority, optional
            Priority level of the task
        task_type : TaskType, optional
            Category of the task
        dependencies : list, optional
            List of task IDs that must complete before this task can start
        group_id : str, optional
            Group ID to add this task to
            
        Returns
        -------
        str
            ID of the created task
        """
        # Generate a unique task ID
        task_id = self._generate_unique_id("task")
        
        # Default empty collections
        kwargs = kwargs or {}
        dependencies = dependencies or []
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(function)
        
        # Create the task
        task = Task(
            task_id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            task_type=task_type,
            dependencies=dependencies.copy(),
            is_async=is_async
        )
        
        # Add task to our tracking dictionary
        self._tasks[task_id] = task
        
        # Add task to group if specified
        if group_id and group_id in self._groups:
            self._groups[group_id].add_task(task_id)
        
        # Check if task has unfinished dependencies
        has_pending_dependency = False
        for dep_id in dependencies:
            if dep_id in self._tasks:
                dep_task = self._tasks[dep_id]
                if dep_task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    has_pending_dependency = True
                    break
        
        # If no pending dependencies, add to queue
        if not has_pending_dependency and task.status != TaskStatus.WAITING_DEPENDENCY:
            self._enqueue_task(task_id)
        
        # Start processing the queue
        self._schedule_tasks()
        
        logger.debug(f"Created task {task_id}: {name} (priority={priority.name}, type={task_type.name})")
        return task_id
    
    def create_task_group(self, name: str) -> str:
        """
        Create a new task group.
        
        Parameters
        ----------
        name : str
            Name of the task group
            
        Returns
        -------
        str
            ID of the created task group
        """
        group_id = self._generate_unique_id("group")
        self._groups[group_id] = TaskGroup(group_id, name)
        logger.debug(f"Created task group {group_id}: {name}")
        return group_id
    
    def add_task_to_group(self, task_id: str, group_id: str) -> bool:
        """
        Add an existing task to a task group.
        
        Parameters
        ----------
        task_id : str
            ID of the task to add
        group_id : str
            ID of the group to add the task to
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if task_id not in self._tasks or group_id not in self._groups:
            return False
        
        self._groups[group_id].add_task(task_id)
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific task by ID.
        
        Parameters
        ----------
        task_id : str
            ID of the task to cancel
            
        Returns
        -------
        bool
            True if task was cancelled, False if not found or already completed
        """
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        # Skip already completed, failed, or cancelled tasks
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        
        # Cancel worker if it's running
        if task.worker is not None:
            task.worker.cancel()
        
        # If task is queued but not started, remove it from queue
        if task.status == TaskStatus.QUEUED:
            # Note: This is inefficient for large queues but working with heapq directly
            # would be more complex. In practice, the queue size should be manageable.
            self._task_queue = [(p, tid) for p, tid in self._task_queue if tid != task_id]
            heapq.heapify(self._task_queue)
        
        # Update status
        task.status = TaskStatus.CANCELLED
        
        # Emit signal
        self.signals.task_cancelled.emit(task_id)
        
        logger.debug(f"Cancelled task {task_id}: {task.name}")
        return True
    
    def cancel_group(self, group_id: str) -> int:
        """
        Cancel all tasks in a group.
        
        Parameters
        ----------
        group_id : str
            ID of the group to cancel
            
        Returns
        -------
        int
            Number of tasks cancelled
        """
        if group_id not in self._groups:
            return 0
        
        group = self._groups[group_id]
        group.is_cancelled = True
        
        cancelled_count = 0
        for task_id in group.task_ids:
            if self.cancel_task(task_id):
                cancelled_count += 1
        
        self.signals.group_cancelled.emit(group_id)
        
        logger.debug(f"Cancelled group {group_id}: {group.name} ({cancelled_count} tasks)")
        return cancelled_count
    
    def cancel_all_tasks(self) -> int:
        """
        Cancel all managed tasks.
        
        Returns
        -------
        int
            Number of tasks cancelled
        """
        cancelled_count = 0
        for task_id in list(self._tasks.keys()):
            if self.cancel_task(task_id):
                cancelled_count += 1
        
        logger.debug(f"Cancelled all tasks ({cancelled_count} total)")
        return cancelled_count
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Get the current status of a task.
        
        Parameters
        ----------
        task_id : str
            ID of the task
            
        Returns
        -------
        TaskStatus
            Current status of the task
            
        Raises
        ------
        KeyError
            If task_id is not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        return self._tasks[task_id].status
    
    def get_task_progress(self, task_id: str) -> float:
        """
        Get the current progress of a task.
        
        Parameters
        ----------
        task_id : str
            ID of the task
            
        Returns
        -------
        float
            Current progress value (0.0 to 1.0)
            
        Raises
        ------
        KeyError
            If task_id is not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        
        if task.status == TaskStatus.COMPLETED:
            return 1.0
        
        return task.progress
    
    def get_group_progress(self, group_id: str) -> float:
        """
        Get the aggregate progress of all tasks in a group.
        
        Parameters
        ----------
        group_id : str
            ID of the group
            
        Returns
        -------
        float
            Average progress of all tasks in the group (0.0 to 1.0)
            
        Raises
        ------
        KeyError
            If group_id is not found
        """
        if group_id not in self._groups:
            raise KeyError(f"Group {group_id} not found")
        
        return self._groups[group_id].get_progress(self._tasks)
    
    def get_task_result(self, task_id: str) -> Any:
        """
        Get the result of a completed task.
        
        Parameters
        ----------
        task_id : str
            ID of the task
            
        Returns
        -------
        Any
            Result of the task if completed, None otherwise
            
        Raises
        ------
        KeyError
            If task_id is not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        
        return None
    
    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """
        Get the error from a failed task.
        
        Parameters
        ----------
        task_id : str
            ID of the task
            
        Returns
        -------
        Optional[Exception]
            Exception from the task if failed, None otherwise
            
        Raises
        ------
        KeyError
            If task_id is not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        
        if task.status == TaskStatus.FAILED:
            return task.error
        
        return None
    
    def get_active_task_count(self) -> int:
        """
        Get the number of currently active (running) tasks.
        
        Returns
        -------
        int
            Number of active tasks
        """
        return self._active_task_count
    
    def get_pending_task_count(self) -> int:
        """
        Get the number of pending (queued but not running) tasks.
        
        Returns
        -------
        int
            Number of pending tasks
        """
        return len(self._task_queue)
    
    def get_total_task_count(self) -> int:
        """
        Get the total number of managed tasks in all states.
        
        Returns
        -------
        int
            Total number of tasks
        """
        return len(self._tasks)
    
    def _schedule_tasks(self) -> None:
        """Internal method to schedule and start tasks from the queue."""
        # While we have tasks in the queue and capacity to run more
        while (
            self._task_queue and 
            self._active_task_count < self._max_concurrent_tasks
        ):
            # Get highest priority task from the queue (lowest number = highest priority)
            _, task_id = heapq.heappop(self._task_queue)
            
            # Task may have been removed from tracking or cancelled
            if task_id not in self._tasks:
                continue
            
            task = self._tasks[task_id]
            
            # Skip tasks that are no longer in a valid state for execution
            if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED, 
                              TaskStatus.FAILED, TaskStatus.CANCELLED):
                continue
            
            # Update task status
            task.status = TaskStatus.RUNNING
            
            # Create appropriate worker based on whether the function is async
            worker = self._create_worker_for_task(task)
            task.worker = worker
            
            # Connect signals
            self._connect_worker_signals(worker, task_id)
            
            # Start worker in thread pool
            self._thread_pool.start(worker)
            
            # Update active task count
            self._active_task_count += 1
            self.signals.active_tasks_changed.emit(self._active_task_count)
            
            logger.debug(f"Started task {task_id}: {task.name} (priority={task.priority.name})")
    
    def _enqueue_task(self, task_id: str) -> None:
        """
        Internal method to add a task to the priority queue.
        
        Parameters
        ----------
        task_id : str
            ID of the task to enqueue
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Update status
        task.status = TaskStatus.QUEUED
        
        # Add to priority queue - note that heapq is a min-heap, so lower values come first
        heapq.heappush(self._task_queue, (task.priority.value, task_id))
        
        logger.debug(f"Enqueued task {task_id}: {task.name} (priority={task.priority.name})")
    
    def _create_worker_for_task(self, task: Task) -> Union[Worker, AsyncWorker]:
        """
        Internal method to create appropriate worker for a task.
        
        Parameters
        ----------
        task : Task
            Task to create worker for
            
        Returns
        -------
        Union[Worker, AsyncWorker]
            Created worker instance
        """
        if task.is_async:
            return AsyncWorker(task.function, *task.args, **task.kwargs)
        else:
            return Worker(task.function, *task.args, **task.kwargs)
    
    def _connect_worker_signals(self, worker: Union[Worker, AsyncWorker], task_id: str) -> None:
        """
        Internal method to connect worker signals to task handler methods.
        
        Parameters
        ----------
        worker : Union[Worker, AsyncWorker]
            Worker to connect signals for
        task_id : str
            ID of the task associated with the worker
        """
        worker.signals.started.connect(lambda: self._on_task_started(task_id))
        worker.signals.result.connect(lambda result: self._on_task_result(task_id, result))
        worker.signals.error.connect(lambda error: self._on_task_error(task_id, error))
        worker.signals.progress.connect(lambda p: self._on_task_progress(task_id, p))
        worker.signals.finished.connect(lambda: self._on_task_finished(task_id))
        worker.signals.cancelled.connect(lambda: self._on_task_cancelled(task_id))
    
    @pyqtSlot()
    def _on_task_started(self, task_id: str) -> None:
        """
        Handler for task started signal.
        
        Parameters
        ----------
        task_id : str
            ID of the started task
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Verify that task is in running state
        if task.status != TaskStatus.RUNNING:
            return
            
        # Emit task started signal
        self.signals.task_started.emit(task_id)
        
        # Check if this is the first task in any groups, and emit group started if so
        for group_id, group in self._groups.items():
            if task_id in group.task_ids:
                # Check if any tasks in the group are already running/completed
                group_has_started = False
                for other_task_id in group.task_ids:
                    if other_task_id != task_id and other_task_id in self._tasks:
                        other_task = self._tasks[other_task_id]
                        if other_task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
                            group_has_started = True
                            break
                
                # If this is the first task in the group to start, emit group started
                if not group_has_started:
                    self.signals.group_started.emit(group_id)
        
        logger.debug(f"Task {task_id} started: {task.name}")
    
    @pyqtSlot(object)
    def _on_task_result(self, task_id: str, result: Any) -> None:
        """
        Handler for task result signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task
        result : Any
            Result data from the task
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Store result
        task.result = result
        
        logger.debug(f"Task {task_id} produced result: {task.name}")
    
    @pyqtSlot(Exception)
    def _on_task_error(self, task_id: str, error: Exception) -> None:
        """
        Handler for task error signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task
        error : Exception
            Error raised by the task
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Update status
        task.status = TaskStatus.FAILED
        
        # Store error
        task.error = error
        
        # Emit error signal
        self.signals.task_failed.emit(task_id, error)
        
        logger.error(f"Task {task_id} failed: {task.name} - Error: {str(error)}")
    
    @pyqtSlot(float)
    def _on_task_progress(self, task_id: str, progress: float) -> None:
        """
        Handler for task progress signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task
        progress : float
            Progress value (0.0 to 1.0)
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Update progress
        task.progress = progress
        
        # Emit progress signal
        self.signals.task_progress.emit(task_id, progress)
        
        # Update progress for any groups this task belongs to
        for group_id, group in self._groups.items():
            if task_id in group.task_ids:
                group_progress = group.get_progress(self._tasks)
                self.signals.group_progress.emit(group_id, group_progress)
        
        # Log progress (debug level)
        logger.debug(f"Task {task_id} progress: {task.name} - {progress:.2f}")
    
    @pyqtSlot()
    def _on_task_finished(self, task_id: str) -> None:
        """
        Handler for task finished signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # If task isn't already in a terminal state, mark as completed
        if task.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.status = TaskStatus.COMPLETED
            
            # Emit completed signal with result
            self.signals.task_completed.emit(task_id, task.result)
        
        # Update active task count
        self._active_task_count -= 1
        self.signals.active_tasks_changed.emit(self._active_task_count)
        
        # Update any tasks that depend on this one
        self._update_dependent_tasks(task_id)
        
        # Check if any groups containing this task are now complete
        self._check_group_completion(task_id)
        
        # Schedule next tasks
        self._schedule_tasks()
        
        logger.debug(f"Task {task_id} finished: {task.name}")
    
    @pyqtSlot()
    def _on_task_cancelled(self, task_id: str) -> None:
        """
        Handler for task cancelled signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task
        """
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        
        # Update status
        task.status = TaskStatus.CANCELLED
        
        # Emit cancelled signal
        self.signals.task_cancelled.emit(task_id)
        
        # Only decrement active count if task was running
        was_running = task.status == TaskStatus.RUNNING
        if was_running:
            self._active_task_count -= 1
            self.signals.active_tasks_changed.emit(self._active_task_count)
        
        # Schedule next tasks
        self._schedule_tasks()
        
        logger.debug(f"Task {task_id} cancelled: {task.name}")
    
    def _update_dependent_tasks(self, completed_task_id: str) -> None:
        """
        Update tasks that depend on a completed task.
        
        Parameters
        ----------
        completed_task_id : str
            ID of the completed task
        """
        # Find tasks that depend on the completed task
        for task_id, task in self._tasks.items():
            if completed_task_id in task.dependencies:
                # Remove the dependency
                task.dependencies.remove(completed_task_id)
                
                # If no remaining dependencies and task is waiting, enqueue it
                if not task.dependencies and task.status == TaskStatus.WAITING_DEPENDENCY:
                    task.status = TaskStatus.PENDING
                    self._enqueue_task(task_id)
                    logger.debug(f"Task {task_id} unblocked by completion of {completed_task_id}")
    
    def _check_group_completion(self, task_id: str) -> None:
        """
        Check if all tasks in a group are completed, failed, or cancelled.
        
        Parameters
        ----------
        task_id : str
            ID of a task to check group completion for
        """
        # Find all groups that contain this task
        for group_id, group in self._groups.items():
            if task_id in group.task_ids:
                # Check if all tasks in the group are in terminal state
                all_done = True
                for other_task_id in group.task_ids:
                    if other_task_id in self._tasks:
                        other_task = self._tasks[other_task_id]
                        if other_task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                            all_done = False
                            break
                
                # If all tasks are done, emit group completed
                if all_done:
                    self.signals.group_completed.emit(group_id)
                    logger.debug(f"Group {group_id} completed: {group.name}")
    
    def _generate_unique_id(self, prefix: str) -> str:
        """
        Generate a unique ID for tasks or groups.
        
        Parameters
        ----------
        prefix : str
            Prefix for the ID (e.g., 'task', 'group')
            
        Returns
        -------
        str
            Unique ID with given prefix
        """
        # Generate a random suffix
        suffix = uuid.uuid4().hex[:8]
        return f"{prefix}-{suffix}"