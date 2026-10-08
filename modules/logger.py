import json
import logging
import sys
from datetime import datetime, timezone


def setup_logger(debug_mode=False, json_mode=False):
    """(Re)configure the shared 'DatasetManager' logger.

    json_mode=True switches the console handler to JSON-lines output for
    machine consumers; the default human format is otherwise unchanged.
    """
    # Get the logger
    logger = logging.getLogger('DatasetManager')

    # If logger already has handlers, remove them (to allow changing debug mode)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set base level based on debug mode
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create console handler only (no file logging). JSONL goes to stdout
    # (the machine stream); the human format stays on stderr so a TTY sees
    # it while scripts can capture stdout alone.
    c_handler = logging.StreamHandler(sys.stdout if json_mode else sys.stderr)
    c_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create formatter
    if json_mode:
        c_formatter = JsonLinesFormatter()
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_formatter = HumanFormatterWithEvents(log_format, datefmt='%H:%M:%S')
    c_handler.setFormatter(c_formatter)

    logger.addHandler(c_handler)

    return logger


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line for machine consumers (AI agents).

    log_event() attaches structured fields via extra={'event_data': {...}};
    plain logger calls still emit valid JSONL with just the message.
    """

    def format(self, record):
        entry = {
            'ts': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
        }
        event_data = getattr(record, 'event_data', None)
        if event_data:
            entry['data'] = event_data
        if record.exc_info:
            entry['exc'] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class HumanFormatterWithEvents(logging.Formatter):
    """The classic console format, plus any event fields appended as JSON."""

    def format(self, record):
        line = super().format(record)
        event_data = getattr(record, 'event_data', None)
        if event_data:
            line += ' ' + json.dumps(event_data, ensure_ascii=False)
        return line


def add_log_file_handler(logger, log_file, level=logging.DEBUG):
    """Mirror all log records to log_file (JSONL when the console is JSONL).

    Returns the added handler so callers can remove it later.
    """
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(level)
    handler.setFormatter(logger.handlers[0].formatter if logger.handlers
                         else logging.Formatter())
    logger.addHandler(handler)
    return handler


def log_event(logger, event, **fields):
    """Emit one structured event; fields land under 'data' in JSONL mode."""
    logger.info(event, extra={'event_data': fields} if fields else None)
