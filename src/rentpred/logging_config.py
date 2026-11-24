import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """Turn every log record into one line of JSON."""

    def format(self, record):
        entry = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        return json.dumps(entry)


def get_logger(name="mp234-api"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger