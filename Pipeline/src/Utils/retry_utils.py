import time
import logging
import functools
from typing import Callable, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable)


def retry(
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    max_attempts: int = 3,
    initial_delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    logger: logging.Logger = None,
) -> Callable[[F], F]:
    """
    Generic retry decorator with exponential backoff, for plain Python
    functions that aren't already wrapped in a Prefect @task (which has its
    own retries= / retry_delay_seconds= built in -- don't stack this on top
    of a Prefect task, use it for helper functions a task calls internally,
    e.g. a single HTTP request inside acquisition.run()).

    Usage:
        @retry(exceptions=(requests.exceptions.RequestException,), max_attempts=3)
        def fetch_one_file(url):
            ...

    Re-raises the last exception if every attempt fails.
    """
    def decorator(func: F) -> F:
        log = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay_seconds
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    log.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_multiplier
            raise last_exception  # pragma: no cover - unreachable safeguard

        return wrapper

    return decorator