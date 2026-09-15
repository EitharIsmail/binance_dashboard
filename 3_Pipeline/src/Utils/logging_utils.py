import sys
import logging


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logging so that every module's `logging.getLogger(__name__)`
    calls actually print somewhere. Call this once from your entry point
    (main.py) -- individual modules should never call basicConfig themselves.

    Note: Prefect's get_run_logger() inside @task-decorated functions is a
    separate logger and works independently of this; this only affects plain
    `logger.info(...)` calls in classes like DataPreprocessor, FeatureEngineer,
    and the Models/ modules.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )