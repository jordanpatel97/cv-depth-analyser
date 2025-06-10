import logging, sys

def configure_logging(level="INFO"):
    lvl = getattr(logging, level) if isinstance(level, str) else level
    logging.basicConfig(stream=sys.stderr,
                        level=lvl,
                        format="%(asctime)s | %(name)-14s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
