import logging
import logging.handlers
import queue
import threading
import os
import atexit
from typing import List, Optional, Union

from global_config.settings import global_settings, LOG_ROOT

# --- Global Queue and Listener ---
# Using a single global queue simplifies setup.
# The listener is also global to manage its lifecycle (start/stop).
log_queue: queue.Queue = queue.Queue()
listener: Optional[logging.handlers.QueueListener] = None

# --- Default Configuration ---
# Sensible defaults for common use cases. Can be overridden via setup_logging.
DEFAULT_LOG_LEVEL: int = logging.INFO if global_settings.LOG_LEVEL == "INFO" else logging.DEBUG
DEFAULT_CONSOLE_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def get_prefect_or_regular_python_logger(name: str, log_level: int = global_settings.LOG_LEVEL, return_non_prefect_logger: bool = True) -> logging.Logger:
    """
    Returns the appropriate logger based on the environment.
    
    In Prefect flow context, returns the Prefect run logger.
    Otherwise, returns a regular Python logger configured with async queue handling.
    
    Args:
        name (str): Logger name (used only for regular Python logger)
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        return_non_prefect_logger (bool): If True, returns regular logger when not in Prefect context.
                                         If False, returns None when not in Prefect context.
    
    Returns:
        logging.Logger: The appropriate logger instance, or None if return_non_prefect_logger is False
                       and not in Prefect context.
    """
    try:
        from prefect.logging import get_run_logger
        from prefect.exceptions import MissingContextError
        logger = get_run_logger()
        logger.setLevel(log_level)
        return logger
    except (ImportError, Exception):
        return get_logger(name) if return_non_prefect_logger else None


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    console_format: str = DEFAULT_CONSOLE_FORMAT,
) -> None:
    """
    Configures asynchronous, non-blocking console logging for the application.

    Uses a QueueHandler to send log records to a background QueueListener
    thread, which then dispatches them to the console handler.
    This prevents logging I/O from blocking the main application threads.

    This setup is optimized for containerized environments (Docker/Kubernetes)
    where console output is captured by the container runtime.

    It's recommended to call this function once at application startup.
    Subsequent calls will stop the existing listener and reconfigure.

    Args:
        log_level (int): The minimum logging level for the root logger
                         (e.g., logging.DEBUG, logging.INFO).
                         Defaults to logging.INFO.
        console_format (str): The format string for console log messages.
    
    Note:
        File logging has been removed to prevent file permission issues
        in containerized environments. All logs are sent to stdout/stderr
        for collection by container logging drivers.
    """
    global listener
    global log_queue # Ensure we are using the global queue

    # --- Create Console Handler ---
    console_formatter = logging.Formatter(console_format)
    
    # Logs to stderr by default (standard for containerized apps)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # --- Configure Queue Handling ---
    # The QueueHandler takes log records and puts them onto the global queue.
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # --- Configure Root Logger ---
    # Get the root logger. Configuring the root logger makes this setup
    # automatically apply to all loggers obtained via logging.getLogger()
    # unless they specifically disable propagation or override handlers.
    root_logger = logging.getLogger()

    # Remove existing handlers attached to the root logger.
    # This is crucial if setup_logging is called multiple times or if
    # other libraries have already configured logging.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our queue handler to the root logger.
    root_logger.addHandler(queue_handler)

    # Set the desired level on the root logger.
    # This determines which messages are even processed and put into the queue.
    root_logger.setLevel(log_level)

    # --- Start the Listener ---
    # Stop the existing listener if setup_logging is called again.
    stop_listener() # Ensure any previous listener is stopped

    # The QueueListener pulls records from log_queue and passes them to
    # the console handler in its own thread.
    # respect_handler_level=True means the listener will check handler levels too.
    listener = logging.handlers.QueueListener(log_queue, console_handler, respect_handler_level=True)
    listener.start()

    # --- Register Shutdown Hook ---
    # Ensure the listener is stopped gracefully on normal Python exit.
    # This allows the queue to be flushed.
    atexit.register(stop_listener)


logging_is_setup = False

def get_logger(
        name: str, 
        log_level: str = global_settings.LOG_LEVEL,
    ) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    This is the standard way application modules should get their logger.
    Since the root logger is configured by setup_logging, loggers obtained
    via this function will automatically inherit the asynchronous setup
    (i.e., their records will go through the QueueHandler).

    All logs are sent to console only, which is ideal for containerized
    environments where logs are collected by Docker/Kubernetes.

    Args:
        name (str): The name for the logger. Typically, use __name__ for
                    the calling module to leverage the logging hierarchy.
        log_level (str): The logging level. Defaults to the global setting.

    Returns:
        logging.Logger: The logger instance configured for async console output.
    """
    global logging_is_setup
    if not logging_is_setup:
        setup_logging(log_level=log_level)
        logging_is_setup = True
    # Simply return the logger. Configuration is handled at the root level.
    return logging.getLogger(name)


def stop_listener() -> None:
    """
    Stops the logging listener thread gracefully.

    This function is registered with atexit to be called on normal exit,
    but can also be called manually if needed (e.g., during specific
    application shutdown phases before atexit runs).
    
    This ensures all queued log messages are flushed before shutdown.
    """
    global listener
    if listener:
        try:
            listener.stop() # This signals the listener thread to stop and waits
            listener = None
        except Exception:
            # Silently ignore errors during shutdown
            pass


# --- Example Usage (Illustrative) ---
# This block demonstrates how to use the setup.
# Typically, you would call setup_logging() once in your application's
# main entry point (e.g., main.py, app factory).
if __name__ == "__main__":
    print("Running logging example...")

    # Configure logging: DEBUG level, console output only
    setup_logging(log_level=logging.DEBUG)

    # Get loggers for different parts of a hypothetical application
    main_logger = get_logger('main_app')
    module_logger = get_logger('main_app.module') # Child logger

    # Log messages from different levels and modules
    main_logger.info("Application starting up...")
    module_logger.debug("Debugging information from the module.")
    main_logger.warning("A potential issue was detected.")
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError:
        main_logger.error("An error occurred during calculation!", exc_info=True) # Log exception info

    module_logger.info("Module finished its task.")
    main_logger.info("Application shutting down...")

    print("Example finished. Check console output for all logs.")
