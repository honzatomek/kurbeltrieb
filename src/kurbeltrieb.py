#!/usr/bin/env python3

import version

__doc__ = f"""Crankshaft and Piston kinematics using SymPy

author:  {version.__author__:s}
date:    {version.__date__:s}
version: {version.__version__:s}

description:
{version.__description__:s}
"""

import os
import sys
import warnings
import argparse
import logging
import datetime

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

import matplotlib
import matplotlib.pyplot as plt

#------------------------------------------------------------- GENERAL SETUP ---#
warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")
PLOT_SIZE = (16, 10)
_NOW = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")

#------------------------------------------------------------- LOGGING SETUP ---#
def addLoggingLevel(levelName, levelNum, methodName=None):
    """from: https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


_LOG_FORMATTER = logging.Formatter(fmt=" %(asctime)13s *%(levelname).1s* %(message)s",
                                   datefmt="%y%m%d-%H%M%S", style="%")
_OFFSET = 21
_NL = "\n" + " " * _OFFSET
# logLines = lambda lines: "\n".join([line if i == 0 else " " * _OFFSET + line for i, line in enumerate(lines)])
logLines = lambda lines: "> " + lines[0] + _NL + _NL.join(lines[1:]) + "\n"

# create new level = MESSAGE
if not hasattr(logging, "MESSAGE"):
    addLoggingLevel("MESSAGE", 100)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# console handler
_LOG_CH = logging.StreamHandler()
_LOG_CH.setLevel(logging.DEBUG)
_LOG_CH.setFormatter(_LOG_FORMATTER)
logger.addHandler(_LOG_CH)


#------------------------------------------------------------- WINDOWS SETUP ---#
if sys.platform == "win32":
    matplotlib.use("Qt5Agg")
else:
    # matplotlib.use("TkAgg")
    pass


#--------------------------------------------------------------- SYMPY SETUP ---#
sm.init_printing(use_latex='mathjax')

class ReferenceFrame(me.ReferenceFrame):
    """set up sympy.physics.mechanics ReferenceFrame module for nicer symbols"""

    def __init__(self, *args, **kwargs):

        kwargs.pop('latexs', None)

        lab = args[0].lower()
        tex = r'\hat{{{}}}_{}'

        super(ReferenceFrame, self).__init__(*args,
                                             latexs=(tex.format(lab, 'x'),
                                                     tex.format(lab, 'y'),
                                                     tex.format(lab, 'z')),
                                             **kwargs)
me.ReferenceFrame = ReferenceFrame




if __name__ == "__main__":
    # Create options parser object.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Add arguments
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="""Increase verbosity.""")

    parser.add_argument("-o", "--output", dest="result_file", nargs=1, type=str, default=[None],
                        help="""Save the results as a text file""")

    # Parse command-line arguments.
    args = parser.parse_args()

    # # set up logging to file
    # logname = os.path.splitext(args.config_file)[0] + "_" + _NOW + ".pro"
    # _LOG_FH = logging.FileHandler(filename=logname, mode="w", encoding="utf-8")
    # _LOG_FH.setLevel(logging.DEBUG)
    # _LOG_FH.setFormatter(_LOG_FORMATTER)
    # logger.addHandler(_LOG_FH)

    # # set logging
    # # level = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    # level = [logging.WARNING, logging.INFO, logging.DEBUG]
    # _LOG_FH.setLevel(level[min(len(level) - 1, args.verbose)])
    # _LOG_CH.setLevel(level[min(len(level) - 1, args.verbose)])

