# central helper — safe to import anywhere
import logging
import os

def setup_logging(default_level="INFO"):
    """Idempotent: only configures if nothing is configured yet."""
    root = logging.getLogger()
    if root.handlers:  # already configured somewhere else
        return

    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", encoding="utf-8"),
        ],
    )
