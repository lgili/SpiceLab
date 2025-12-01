"""Hook system for plugin event handling.

This module provides a flexible hook system for plugins to intercept
and modify SpiceLab behavior at various points.

Example::

    from spicelab.plugins import HookManager, HookType, HookPriority

    # Register a hook function
    @HookManager.register(HookType.PRE_SIMULATION)
    def log_simulation(circuit, analyses):
        print(f"Simulating {circuit.name} with {len(analyses)} analyses")

    # Register with priority
    @HookManager.register(HookType.POST_SIMULATION, priority=HookPriority.HIGH)
    def cache_results(circuit, result):
        cache.store(circuit.hash(), result)

    # Manually trigger hooks
    HookManager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=analyses)

"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type for hook callbacks
HookCallback = Callable[..., Any]
F = TypeVar("F", bound=Callable[..., Any])


class HookType(Enum):
    """Types of hooks available in SpiceLab.

    Hooks are organized by the phase of operation they intercept.
    """

    # Circuit building hooks
    PRE_COMPONENT_ADD = auto()  # Before adding a component
    POST_COMPONENT_ADD = auto()  # After adding a component
    PRE_COMPONENT_REMOVE = auto()  # Before removing a component
    POST_COMPONENT_REMOVE = auto()  # After removing a component
    PRE_NETLIST_BUILD = auto()  # Before building netlist
    POST_NETLIST_BUILD = auto()  # After building netlist

    # Simulation hooks
    PRE_SIMULATION = auto()  # Before running simulation
    POST_SIMULATION = auto()  # After simulation completes
    SIMULATION_ERROR = auto()  # When simulation fails
    SIMULATION_PROGRESS = auto()  # During simulation (for progress)

    # Analysis hooks
    PRE_ANALYSIS = auto()  # Before running analysis
    POST_ANALYSIS = auto()  # After analysis completes

    # Validation hooks
    PRE_VALIDATION = auto()  # Before circuit validation
    POST_VALIDATION = auto()  # After circuit validation
    VALIDATION_ERROR = auto()  # When validation fails

    # Measurement hooks
    PRE_MEASUREMENT = auto()  # Before taking measurement
    POST_MEASUREMENT = auto()  # After measurement completes

    # Result hooks
    RESULT_CREATED = auto()  # When result object is created
    RESULT_EXPORTED = auto()  # When result is exported

    # Plugin lifecycle hooks
    PLUGIN_LOADED = auto()  # When a plugin is loaded
    PLUGIN_ACTIVATED = auto()  # When a plugin is activated
    PLUGIN_DEACTIVATED = auto()  # When a plugin is deactivated
    PLUGIN_ERROR = auto()  # When a plugin encounters an error

    # Engine hooks
    ENGINE_REGISTERED = auto()  # When an engine is registered
    ENGINE_SELECTED = auto()  # When an engine is selected

    # Cache hooks
    CACHE_HIT = auto()  # When cache hit occurs
    CACHE_MISS = auto()  # When cache miss occurs
    CACHE_STORE = auto()  # When storing in cache


class HookPriority(Enum):
    """Priority levels for hook execution order.

    Higher priority hooks are executed first.
    """

    CRITICAL = 100  # Execute first (e.g., security checks)
    HIGH = 75  # Execute early
    NORMAL = 50  # Default priority
    LOW = 25  # Execute late
    LOWEST = 0  # Execute last (e.g., cleanup)


@dataclass
class Hook:
    """Represents a registered hook.

    Attributes:
        hook_type: Type of hook
        callback: Function to call
        priority: Execution priority
        plugin_name: Name of plugin that registered this hook
        enabled: Whether hook is currently enabled
        description: Human-readable description
    """

    hook_type: HookType
    callback: HookCallback
    priority: HookPriority = HookPriority.NORMAL
    plugin_name: str = ""
    enabled: bool = True
    description: str = ""
    _id: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self._id:
            self._id = id(self.callback)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook callback."""
        if self.enabled:
            return self.callback(*args, **kwargs)
        return None

    def __hash__(self) -> int:
        return self._id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Hook):
            return self._id == other._id
        return False


class HookManager:
    """Central manager for all hooks in SpiceLab.

    This is a singleton class that manages hook registration and execution.
    Hooks are executed in priority order (highest first).

    Thread Safety:
        All operations are thread-safe using internal locks.

    Example::

        # Register a hook
        HookManager.register_hook(
            HookType.PRE_SIMULATION,
            my_callback,
            priority=HookPriority.HIGH,
            plugin_name="my-plugin"
        )

        # Trigger hooks
        results = HookManager.trigger(
            HookType.PRE_SIMULATION,
            circuit=circuit,
            analyses=analyses
        )

        # Unregister
        HookManager.unregister_hook(HookType.PRE_SIMULATION, my_callback)
    """

    _instance: HookManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> HookManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize the hook manager."""
        self._hooks: dict[HookType, list[Hook]] = {t: [] for t in HookType}
        self._hooks_lock = threading.Lock()
        self._enabled = True

    @classmethod
    def get_instance(cls) -> HookManager:
        """Get the singleton instance."""
        return cls()

    @classmethod
    def register(
        cls,
        hook_type: HookType,
        priority: HookPriority = HookPriority.NORMAL,
        plugin_name: str = "",
        description: str = "",
    ) -> Callable[[F], F]:
        """Decorator to register a hook function.

        Args:
            hook_type: Type of hook to register
            priority: Execution priority
            plugin_name: Name of plugin registering the hook
            description: Human-readable description

        Returns:
            Decorator function

        Example::

            @HookManager.register(HookType.PRE_SIMULATION)
            def my_hook(circuit, analyses):
                print("About to simulate!")
        """

        def decorator(func: F) -> F:
            instance = cls.get_instance()
            instance.register_hook(
                hook_type=hook_type,
                callback=func,
                priority=priority,
                plugin_name=plugin_name,
                description=description or func.__doc__ or "",
            )
            return func

        return decorator

    def register_hook(
        self,
        hook_type: HookType,
        callback: HookCallback,
        priority: HookPriority = HookPriority.NORMAL,
        plugin_name: str = "",
        description: str = "",
    ) -> Hook:
        """Register a hook callback.

        Args:
            hook_type: Type of hook
            callback: Function to call
            priority: Execution priority
            plugin_name: Name of plugin registering this hook
            description: Human-readable description

        Returns:
            The registered Hook object
        """
        hook = Hook(
            hook_type=hook_type,
            callback=callback,
            priority=priority,
            plugin_name=plugin_name,
            description=description,
        )

        with self._hooks_lock:
            self._hooks[hook_type].append(hook)
            # Sort by priority (descending)
            self._hooks[hook_type].sort(key=lambda h: h.priority.value, reverse=True)

        logger.debug(
            f"Registered hook: {hook_type.name} from {plugin_name or 'anonymous'} "
            f"with priority {priority.name}"
        )
        return hook

    def unregister_hook(
        self,
        hook_type: HookType,
        callback: HookCallback,
    ) -> bool:
        """Unregister a hook callback.

        Args:
            hook_type: Type of hook
            callback: The callback to unregister

        Returns:
            True if hook was found and removed
        """
        with self._hooks_lock:
            hooks = self._hooks[hook_type]
            for i, hook in enumerate(hooks):
                if hook.callback is callback or id(hook.callback) == id(callback):
                    hooks.pop(i)
                    logger.debug(f"Unregistered hook: {hook_type.name}")
                    return True
        return False

    def unregister_plugin_hooks(self, plugin_name: str) -> int:
        """Unregister all hooks from a plugin.

        Args:
            plugin_name: Name of plugin

        Returns:
            Number of hooks removed
        """
        count = 0
        with self._hooks_lock:
            for hook_type in HookType:
                original_len = len(self._hooks[hook_type])
                self._hooks[hook_type] = [
                    h for h in self._hooks[hook_type] if h.plugin_name != plugin_name
                ]
                count += original_len - len(self._hooks[hook_type])
        if count:
            logger.debug(f"Unregistered {count} hooks from plugin {plugin_name}")
        return count

    @classmethod
    def trigger(
        cls,
        hook_type: HookType,
        *args: Any,
        stop_on_error: bool = False,
        stop_on_false: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """Trigger all hooks of a given type.

        Args:
            hook_type: Type of hooks to trigger
            *args: Positional arguments to pass to hooks
            stop_on_error: Stop execution if a hook raises an exception
            stop_on_false: Stop execution if a hook returns False
            **kwargs: Keyword arguments to pass to hooks

        Returns:
            List of return values from all hooks

        Raises:
            Exception: If stop_on_error is True and a hook raises
        """
        instance = cls.get_instance()
        return instance._trigger(
            hook_type,
            *args,
            stop_on_error=stop_on_error,
            stop_on_false=stop_on_false,
            **kwargs,
        )

    def _trigger(
        self,
        hook_type: HookType,
        *args: Any,
        stop_on_error: bool = False,
        stop_on_false: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """Internal trigger implementation."""
        if not self._enabled:
            return []

        results = []
        with self._hooks_lock:
            hooks = list(self._hooks[hook_type])  # Copy to avoid lock during execution

        for hook in hooks:
            if not hook.enabled:
                continue
            try:
                result = hook(*args, **kwargs)
                results.append(result)
                if stop_on_false and result is False:
                    break
            except Exception as e:
                logger.error(f"Hook {hook.callback.__name__} ({hook_type.name}) raised: {e}")
                if stop_on_error:
                    raise
                results.append(None)

        return results

    @classmethod
    def trigger_async(
        cls,
        hook_type: HookType,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Trigger hooks asynchronously in a separate thread.

        Args:
            hook_type: Type of hooks to trigger
            *args: Positional arguments to pass to hooks
            **kwargs: Keyword arguments to pass to hooks
        """
        import threading

        thread = threading.Thread(
            target=cls.trigger,
            args=(hook_type,) + args,
            kwargs=kwargs,
            daemon=True,
        )
        thread.start()

    def get_hooks(self, hook_type: HookType) -> list[Hook]:
        """Get all hooks of a given type.

        Args:
            hook_type: Type of hooks to get

        Returns:
            List of Hook objects
        """
        with self._hooks_lock:
            return list(self._hooks[hook_type])

    def get_all_hooks(self) -> dict[HookType, list[Hook]]:
        """Get all registered hooks.

        Returns:
            Dictionary mapping hook types to lists of hooks
        """
        with self._hooks_lock:
            return {k: list(v) for k, v in self._hooks.items()}

    def count_hooks(self, hook_type: HookType | None = None) -> int:
        """Count registered hooks.

        Args:
            hook_type: Optional type to count, or all if None

        Returns:
            Number of hooks
        """
        with self._hooks_lock:
            if hook_type:
                return len(self._hooks[hook_type])
            return sum(len(hooks) for hooks in self._hooks.values())

    def enable(self) -> None:
        """Enable hook execution."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook execution (hooks won't be triggered)."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if hooks are enabled."""
        return self._enabled

    def clear(self, hook_type: HookType | None = None) -> int:
        """Clear all hooks.

        Args:
            hook_type: Optional type to clear, or all if None

        Returns:
            Number of hooks cleared
        """
        count = 0
        with self._hooks_lock:
            if hook_type:
                count = len(self._hooks[hook_type])
                self._hooks[hook_type] = []
            else:
                count = sum(len(hooks) for hooks in self._hooks.values())
                for t in HookType:
                    self._hooks[t] = []
        return count

    def get_info(self) -> dict[str, Any]:
        """Get information about registered hooks."""
        with self._hooks_lock:
            hooks_by_type = {}
            for hook_type, hooks in self._hooks.items():
                if hooks:
                    hooks_by_type[hook_type.name] = [
                        {
                            "callback": h.callback.__name__,
                            "priority": h.priority.name,
                            "plugin": h.plugin_name,
                            "enabled": h.enabled,
                            "description": h.description,
                        }
                        for h in hooks
                    ]

            return {
                "enabled": self._enabled,
                "total_hooks": self.count_hooks(),
                "hooks_by_type": hooks_by_type,
            }

    @classmethod
    def reset(cls) -> None:
        """Reset the hook manager (mainly for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.clear()
                cls._instance._enabled = True


# Context manager for temporarily disabling hooks
class DisableHooks:
    """Context manager to temporarily disable hooks.

    Example::

        with DisableHooks():
            # Hooks won't fire in this block
            circuit.add_component(...)
    """

    def __enter__(self) -> DisableHooks:
        self._manager = HookManager.get_instance()
        self._was_enabled = self._manager.is_enabled()
        self._manager.disable()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._was_enabled:
            self._manager.enable()


# Context manager for temporarily enabling specific hooks
class EnableOnlyHooks:
    """Context manager to temporarily enable only specific hook types.

    Example::

        with EnableOnlyHooks(HookType.PRE_SIMULATION, HookType.POST_SIMULATION):
            # Only simulation hooks will fire
            sim.run(circuit)
    """

    def __init__(self, *hook_types: HookType) -> None:
        self._hook_types = set(hook_types)
        self._manager = HookManager.get_instance()
        self._original_enabled: dict[HookType, list[tuple[Hook, bool]]] = {}

    def __enter__(self) -> EnableOnlyHooks:
        # Disable all hooks except specified types
        for hook_type in HookType:
            hooks = self._manager.get_hooks(hook_type)
            self._original_enabled[hook_type] = [(h, h.enabled) for h in hooks]
            if hook_type not in self._hook_types:
                for hook in hooks:
                    hook.enabled = False
        return self

    def __exit__(self, *args: Any) -> None:
        # Restore original enabled states
        for _hook_type, hook_states in self._original_enabled.items():
            for hook, was_enabled in hook_states:
                hook.enabled = was_enabled
