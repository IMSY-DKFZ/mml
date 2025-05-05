# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import functools
import inspect
import warnings
from _thread import interrupt_main
from threading import Timer
from typing import Any, Callable, Type, Union

from deprecated.sphinx import deprecated, versionadded, versionchanged

__all__ = ["deprecated", "versionadded", "versionchanged", "beta", "timeout"]


def beta(message: str = "BETA:"):
    """
    The beta decorator is used to indicate that a particular feature is in Beta. A callable or type that has
    been marked as beta will give a ``UserWarning`` when it is called or instantiated. Adopted from
    :func:`~flash.core.utilities.stability.beta`.

    :param str message: The message to include in the warning.
    """

    def decorator(_callable: Union[Callable, Type]):
        # if called on a class, recursively call on the class init method
        if inspect.isclass(_callable):
            _callable.__init__ = decorator(_callable.__init__)
            return _callable

        @functools.wraps(_callable)
        def wrapper(*args, **kwargs):
            _raise_beta_warning(message, _callable.__qualname__)
            return _callable(*args, **kwargs)

        return wrapper

    return decorator


@functools.lru_cache()
def _raise_beta_warning(message: str, source: str):
    # lru_cache decorator is used to only warn once for each message / obj
    warnings.warn(
        f"{message} The API and functionality of {source} may change without warning in future releases.",
        category=UserWarning,
    )


def _signal_main(*args, **kwargs) -> None:
    """This function is executed on the Timer thread to signal the timeout back to the main process."""
    interrupt_main()


def timeout(seconds: int) -> Callable:
    """
    A decorator to wrap functions that shall have a timeout attached. Will raise TimeoutError if the execution
    takes longer than the specified number of seconds.

    :param int seconds: timeout in seconds
    :return: the wrapped function
    """

    def decorator(_callable: Callable) -> Callable:
        @functools.wraps(_callable)
        def wrapper(*args, **kwargs) -> Any:
            # set up timer
            timer = Timer(seconds, _signal_main)
            timer.start()
            # invoke underlying function
            try:
                value = _callable(*args, **kwargs)
            # re-interpret the interruption as timeout
            except KeyboardInterrupt:
                if timer.finished.is_set():
                    raise TimeoutError(
                        f"Execution of {_callable.__qualname__} took to long (limit: {seconds} seconds)."
                    )
                else:
                    raise
            # make sure to delete timer anyway
            finally:
                timer.cancel()
            return value

        return wrapper

    return decorator
