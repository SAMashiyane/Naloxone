# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

import logging
import os
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Optional, Union

try:
    from wurlitzer import pipes
except ImportError:
    #  See https://github.com/SAMashiyane/Naloxone
    pipes = None


# https://github.com/SAMashiyane/Naloxone
class LoggerWriter:
    """Writer allowing redirection of streams to logger methods."""

    def __init__(self, writer: Callable):
        self._writer = writer
        self._msg = ""

    def write(self, message: str):
        self._msg = self._msg + message
        while "\n" in self._msg:
            pos = self._msg.find("\n")
            self._writer(self._msg[:pos])
            self._msg = self._msg[pos + 1 :]

    def flush(self):
        if self._msg != "":
            self._writer(self._msg)
            self._msg = ""


class redirect_output:


    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        # Current Python process redirects
        self.redirect_stdout = redirect_stdout(LoggerWriter(self.logger.info))
        self.redirect_stderr = redirect_stderr(LoggerWriter(self.logger.warning))

        self.c_redirect = (
            pipes(
                stdout=LoggerWriter(self.logger.info),
                # stderr=LoggerWriter(self.logger.warning),
            )
            if pipes
            else None
        )

    def __enter__(self):
        self.redirect_stdout.__enter__()
        self.redirect_stderr.__enter__()
        if self.c_redirect:
            self.c_redirect.__enter__()

    def __exit__(self, *args, **kwargs):
        if self.c_redirect:
            self.c_redirect.__exit__(*args, **kwargs)
        self.redirect_stderr.__exit__(*args, **kwargs)
        self.redirect_stdout.__exit__(*args, **kwargs)



def get_logger() -> logging.Logger:
    try:
        assert bool(LOGGER)
        return LOGGER
    except Exception:
        return create_logger(True)


def create_logger(
    log: Union[bool, str, logging.Logger] = True
) -> Union[logging.Logger]:

    if not log:
        return None
    elif isinstance(log, logging.Logger):
        return log

    logger = logging.getLogger("logs")
    level = os.getenv("naloxlib_CUSTOM_LOGGING_LEVEL", "DEBUG")
    logger.setLevel(level)

    logger.propagate = False


    if logger.hasHandlers():
        logger.handlers.clear()


    logPath = os.getenv("naloxlib_CUSTOM_LOGGING_PATH", "Mohtarami.log")
    path = logPath if isinstance(log, bool) else log
    ch: Optional[Union[logging.FileHandler, logging.NullHandler]] = None
    try:
        ch = logging.FileHandler(path)
    except Exception:
        warnings.warn(
            f"Could not attach a FileHandler to the logger at path {path}! "
            "No logs will be saved."
        )
        traceback.print_exc()
        ch = logging.NullHandler()
    ch.setLevel(logging.DEBUG)

    
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")


    ch.setFormatter(formatter)


    logger.addHandler(ch)

    return logger


LOGGER = create_logger()
_warnings_showwarning = None



