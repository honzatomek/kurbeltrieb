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

import scipy

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.animation as animation

from default import default_kurbeltrieb

#------------------------------------------------------------- GENERAL SETUP ---#
warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")
PLOT_SIZE = (16, 10)
_NOW = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")


ZORDER = {"wange":   0,
          "kolben": 10,
          "pleuel": 30,
          "bolzen": 20,}

STEPS_PER_ROTATION = 101

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

    def angleXY(self, B):
        """computes the angle between two reference frames in plane XY"""
        x = self.x.dot(B.x)
        y = self.x.dot(B.y)
        return sm.atan2(y, x)

    def angleXZ(self, B):
        """computes the angle between two reference frames in plane XZ"""
        x = self.x.dot(B.x)
        y = self.x.dot(B.z)
        return -sm.atan2(y, x)

    def angleYZ(self, B):
        """computes the angle between two reference frames in plane YZ"""
        x = self.y.dot(B.y)
        y = self.y.dot(B.z)
        return sm.atan2(y, x)


me.ReferenceFrame = ReferenceFrame


def angleXY(A: me.ReferenceFrame, B: me.ReferenceFrame):
    x = B.x.dot(A.x)
    y = B.x.dot(A.y)
    return sm.atan2(y, x)


#--------------------------------------------------------------- KURBELTRIEB ---#
class KurbeltriebError(Exception):
    pass


def concat(x, y, a):
    x = np.array(x).flatten().astype(float)
    y = np.array(y).flatten().astype(float)
    a = np.array(a).flatten().astype(float)
    maxd = np.max([x.shape[0], y.shape[0], a.shape[0]])
    x = x if x.shape[0] == maxd else np.repeat(x, maxd).flatten()
    y = y if y.shape[0] == maxd else np.repeat(y, maxd).flatten()
    a = a if a.shape[0] == maxd else np.repeat(a, maxd).flatten()
    arr = np.vstack([x, y, a]).T
    return arr


class Part:
    def __init__(self, name: str, body: me.Body, inertia_frame: me.ReferenceFrame):
        self.name = name
        self.body = body
        self.N = inertia_frame

        self._p = None

        self._lambdified = False
        self._pos = None
        self._vel = None
        self._acc = None

        self.ax = None
        self.plt_obj = None
        self.points = None
        self.plot_props = {}

    def set_plotting(self, ax, plot_props, **kwargs):
        self.ax = ax
        self.plot_props = plot_props
        self.points = np.array([0., 0.])

    # def plot(self, coor: np.ndarray, alpha: float):
    def plot(self, a: float, u: float, q: float, p: list = None):
        pos = self.pos(q, u, q, p)
        coor  = pos[0, :2]
        alpha = pos[0,  2]
        T = np.array([[ np.cos(alpha), np.sin(alpha)],
                      [-np.sin(alpha), np.cos(alpha)]], dtype=float)
        if self.plt_obj is None:
            self.plt_obj = plt.Polygon(self.points @ T + coor, **self.plot_props)
            self.ax.add_artist(self.plt_obj)
        else:
            self.plt_obj.set_xy(self.points @ T + coor)

    @property
    def body(self) -> me.Body:
        return self._body

    @body.setter
    def body(self, rigid_body: me.Body):
        self._body = rigid_body

    @property
    def masscenter(self):
        return self._body.masscenter

    @property
    def frame(self):
        return self._body.frame

    def lambdify(self, a, u, q, p: list, repl: dict = {}, dependent: dict = {}):
        logger.debug(f"Lambdifying {self.name:s} equations for position, velocity and acceleration.")
        pos = self.masscenter.pos_from(self.N.masscenter)
        pos = sm.Matrix([pos.dot(self.N.frame.x),
                         pos.dot(self.N.frame.y),
                         self.frame.angleXY(self.N.frame)])

        vel = self.masscenter.vel(self.N.frame)
        vel = sm.Matrix([vel.dot(self.N.frame.x),
                         vel.dot(self.N.frame.y),
                         self.frame.ang_vel_in(self.N.frame).dot(self.N.frame.z)])

        acc = self.masscenter.acc(self.N.frame)
        acc = sm.Matrix([acc.dot(self.N.frame.x),
                         acc.dot(self.N.frame.y),
                         self.frame.ang_acc_in(self.N.frame).dot(self.N.frame.z)])

        for i in reversed(range(3)):
            pos = pos.xreplace(repl[i]).xreplace(dependent[i])
            vel = vel.xreplace(repl[i]).xreplace(dependent[i])
            acc = acc.xreplace(repl[i]).xreplace(dependent[i])

        self._pos = [sm.lambdify((a, u, q, p), pos[0]), sm.lambdify((a, u, q, p), pos[1]), sm.lambdify((a, u, q, p), pos[2])]
        self._vel = [sm.lambdify((a, u, q, p), vel[0]), sm.lambdify((a, u, q, p), vel[1]), sm.lambdify((a, u, q, p), vel[2])]
        self._acc = [sm.lambdify((a, u, q, p), acc[0]), sm.lambdify((a, u, q, p), acc[1]), sm.lambdify((a, u, q, p), acc[2])]

        self._lambdified = True

    @property
    def lambdified(self) -> bool:
        return self._lambdified

    def set_p(self, pvals: list):
        logger.debug(f"Setting parametric values for {self.name:s}.")
        self._p = pvals

    def pos(self, a: float = 0., u: float = 0., q: float = 0., p: list = None) -> np.ndarray:
        if not self.lambdified:
            message = f"{type(self).__name__:s}: the equations must be lambdified first."
            logger.error(message)
            raise KurbeltriebError(message)
        elif p is None:
            if self._p is not None:
                p = self._p
            else:
                message = f"{type(self).__name__:s}: the parametric values must be supplied."
                logger.error(message)
                raise KurbeltriebError(message)
        x, y, a = self._pos[0](a, u, q, p), self._pos[1](a, u, q, p), self._pos[2](a, u, q, p)
        pos = concat(x, y, a)
        return pos

    def vel(self, a: float = 0., u: float = 0., q: float = 0., p: list = None) -> np.ndarray:
        if not self.lambdified:
            message = f"{type(self).__name__:s}: the equations must be lambdified first."
            logger.error(message)
            raise KurbeltriebError(message)
        elif p is None:
            if self._p is not None:
                p = self._p
            else:
                message = f"{type(self).__name__:s}: the parametric values must be supplied."
                logger.error(message)
                raise KurbeltriebError(message)
        x, y, a = self._vel[0](a, u, q, p), self._vel[1](a, u, q, p), self._vel[2](a, u, q, p)
        vel = concat(x, y, a)
        return vel

    def acc(self, a: float = 0., u: float = 0., q: float = 0., p: list = None) -> np.ndarray:
        if not self.lambdified:
            message = f"{type(self).__name__:s}: the equations must be lambdified first."
            logger.error(message)
            raise KurbeltriebError(message)
        elif p is None:
            if self._p is not None:
                p = self._p
            else:
                message = f"{type(self).__name__:s}: the parametric values must be supplied."
                logger.error(message)
                raise KurbeltriebError(message)
        x, y, a = self._acc[0](a, u, q, p), self._acc[1](a, u, q, p), self._acc[2](a, u, q, p)
        acc = concat(x, y, a)
        return acc



class Kolben(Part):
    def set_plotting(self, ax, w: float = 54.0, h: float = 50., plot_props: dict = None):
        _plot_props = {"closed":               True,
                       "fill":                 True,
                       "edgecolor":         "black",
                       "facecolor":         "black",
                       "zorder":    ZORDER["kolben"]}
        if plot_props is not None:
            for k, v in plot_props.items():
                _plot_props[k] = v
        super().set_plotting(ax, _plot_props)

        self.points = np.array([[      -w / 2.,        -h / 2.],
                                [      -w / 2.,         h / 2.],
                                [       w / 2.,         h / 2.],
                                [       w / 2., - 0.5 * h / 2.],
                                [-0.5 * w / 2.,        -h / 2.]], dtype=float)
        self.points += np.array([0., h / 6], dtype=float)



class Wange(Part):
    def set_plotting(self, ax, r: float = 20.00, plot_props: dict = None):
        _plot_props = {"closed":              True,
                       "fill":                True,
                       "edgecolor":        "black",
                       "facecolor":        "black",
                       "zorder":    ZORDER["wange"]}
        if plot_props is not None:
            for k, v in plot_props.items():
                _plot_props[k] = v
        super().set_plotting(ax, _plot_props)

        h = 2 * r
        w = 2 * h
        points = []

        num = 6
        da = (np.pi /  2) / (num - 1)
        a = (np.pi / 2) + da
        for i in range(num):
            a -= da
            points.append([h / 3 * np.cos(a), h / 2 + h / 3 * np.sin(a)])

        points.append([   h / 3, -0.05 * h])
        points.append([ 0.5 * w, -0.05 * h])
        points.append([ 0.5 * w, -0.10 * h])

        r = (points[-1][0] ** 2. + points[-1][1] ** 2) ** 0.5
        a = np.arctan2(points[-1][1], points[-1][0])
        da = (-(np.pi / 2) - a) / (num - 1)
        for i in range(num - 1):
            a += da
            points.append([r * np.cos(a), r * np.sin(a)])

        for point in reversed(points[1:-1]):
            points.append([-point[0], point[1]])

        self.points = np.array(points, dtype=float)



class Pleuel(Part):
    def set_plotting(self, ax, l: float = 71.4, w: float = 8., e: float = 31.159, plot_props: dict = None):
        _plot_props = {"closed":               True,
                       "fill":                 True,
                       "edgecolor":          "grey",
                       "facecolor":          "grey",
                       "zorder":    ZORDER["pleuel"]}
        if plot_props is not None:
            for k, v in plot_props.items():
                _plot_props[k] = v
        super().set_plotting(ax, _plot_props)
        points = []

        num = 6
        da = (np.pi /  2) / (num - 1)
        a = np.pi - da
        for i in range(num):
            a += da
            points.append([w / 2 * np.cos(a), w / 2 * np.sin(a)])

        for point in reversed(points):
            points.append([l - point[0], point[1]])

        for point in reversed(points[1:-1]):
            points.append([point[0], -point[1]])

        offset = np.array([e, 0.], dtype=float)
        points -= offset

        self.points = np.array(points, dtype=float)



class Pin(Part):
    def set_plotting(self, ax, d: float = 10., plot_props: dict = None):
        _plot_props = {"closed":               True,
                       "fill":                 True,
                       "edgecolor":         "white",
                       "facecolor":         "white",
                       "zorder":    ZORDER["bolzen"]}
        if plot_props is not None:
            for k, v in plot_props.items():
                _plot_props[k] = v
        super().set_plotting(ax, _plot_props)
        points = []

        num = 36
        da = 2 * np.pi / (num - 1)
        a = 0. - da
        for i in range(num):
            a += da
            points.append([d / 2 * np.cos(a), d / 2 * np.sin(a)])

        self.points = np.array(points, dtype=float)



class Kurbeltrieb:
    def __init__(self):
        logger.info("Setting Kurbeltrieb analytical model.")
        self.pdict = None
        self.pvals = None

        self._setup = False
        self._lambdified = False
        self._p_set = False

        logger.debug("Setting time independent variables.")
        dir                    = sm.symbols("dir")
        rKUW, eKUW, mKUW, JKUW = sm.symbols("r_{KUW}, e_{KUW}, m_{KUW}, J_{KUW}")
        rUNW, mUNW, aUNW       = sm.symbols("r_{UNW}, m_{UNW}, alpha_{UNW}")
        lPLE, ePLE, mPLE, JPLE = sm.symbols("l_{PLE}, e_{PLE}, m_{PLE}, J_{PLE}")
        mKOL, JKOL             = sm.symbols("m_{KOL}, J_{KOL}")

        self.p = [dir, rKUW, eKUW, mKUW, JKUW, rUNW, mUNW, aUNW, lPLE, ePLE, mPLE, JPLE, mKOL, JKOL]
        self.pnames = {"dir":   dir,
                       "rKUW": rKUW,
                       "eKUW": eKUW,
                       "mKUW": mKUW,
                       "JKUW": JKUW,
                       "rUNW": rUNW,
                       "mUNW": mUNW,
                       "aUNW": aUNW,
                       "lPLE": lPLE,
                       "ePLE": ePLE,
                       "mPLE": mPLE,
                       "JPLE": JPLE,
                       "mKOL": mKOL,
                       "JKOL": JKOL}

        logger.debug("Setting generalised time-dependent variables.")
        qkuw, qple = me.dynamicsymbols("q_{kuw}, q_{ple}")
        ukuw, uple = me.dynamicsymbols("u_{kuw}, u_{ple}")
        akuw, aple = me.dynamicsymbols("a_{kuw}, a_{ple}")

        # independent only
        self.q = qkuw
        self.u = ukuw
        self.a = akuw

        t = me.dynamicsymbols._t

        logger.debug("Setting Inertia Reference frame and the axis of rotation.")
        N = me.Body("O", frame=me.ReferenceFrame("N"))  # inertia reference frame
        self.N = N

        logger.debug("Setting Kurbelwelle.")
        KUW = me.Body("KUW", frame=N.frame.orientnew("KUW_{ref}", "Axis", (dir * qkuw, N.frame.z)))
        KUW.masscenter.central_inertia = me.inertia(KUW.frame, 0, 0, JKUW)
        KUW.frame.set_ang_vel(N.frame, dir * ukuw * N.frame.z)
        KUW.masscenter.set_pos(N.masscenter, 0.)
        KUW.masscenter.set_vel(N.frame, 0.)
        self.KUW = Wange("Kurbelwelle", KUW, N)

        logger.debug("Setting Wange.")
        WNG = me.Body("WNG", mass=mKUW, frame=KUW.frame)
        WNG.masscenter.set_pos(KUW.masscenter, -eKUW * KUW.frame.y)
        WNG.masscenter.v2pt_theory(WNG.masscenter, N.frame, KUW.frame)
        self.WNG = Pin("Wange", WNG, N)

        logger.debug("Setting Unwucht.")
        UNW = me.Body("UNW", mass=mUNW, frame=KUW.frame.orientnew("UNW_{ref}", "Axis", (sm.pi / 2 + aUNW, KUW.frame.z)))
        UNW.masscenter.set_pos(KUW.masscenter, rUNW * UNW.frame.x)
        UNW.masscenter.v2pt_theory(KUW.masscenter, N.frame, KUW.frame)
        self.UNW = Pin("Unwucht", UNW, N)

        logger.debug("Setting Hubzapfen.")
        HZP = me.Body("HZP", frame=N.frame)
        HZP.masscenter.set_pos(KUW.masscenter, rKUW * KUW.frame.y)
        HZP.masscenter.v2pt_theory(KUW.masscenter, N.frame, KUW.frame)
        self.HZP = Pin("Hubzapfen", HZP, N)

        logger.debug("Setting Pleuel.")
        PLE = me.Body("PLE", frame=N.frame.orientnew("PLE_{ref}", "Axis", (qple, N.frame.z)))
        PLE.masscenter.central_inertia = me.inertia(PLE.frame, 0, 0, JPLE)
        PLE.frame.set_ang_vel(N.frame, uple * N.frame.z)
        PLE.masscenter.set_pos(HZP.masscenter, ePLE * PLE.frame.x)
        PLE.masscenter.v2pt_theory(HZP.masscenter, N.frame, PLE.frame)
        self.PLE = Pleuel("Pleuel", PLE, N)

        logger.debug("Setting Bolzen.")
        BLZ = me.Body("BLZ", frame=N.frame)
        BLZ.masscenter.set_pos(HZP.masscenter, lPLE * PLE.frame.x)
        BLZ.masscenter.v2pt_theory(HZP.masscenter, N.frame, PLE.frame)
        self.BLZ = Pin("Bolzen", BLZ, N)

        logger.debug("Setting Kolben.")
        KOL = me.Body("KOL", mass=mKOL, frame=N.frame)
        KOL.masscenter.central_inertia = me.inertia(KOL.frame, 0, 0, JKOL)
        KOL.masscenter.set_pos(BLZ.masscenter, 0.)
        KOL.masscenter.v2pt_theory(BLZ.masscenter, N.frame, PLE.frame)
        self.KOL = Kolben("Kolben", KOL, N)

        logger.debug("Generalised Derivatives replacements.")
        qd_repl  = {qkuw.diff(t): ukuw, qple.diff(t): uple}
        qdd_repl = {qkuw.diff(t, 2): ukuw.diff(t), qple.diff(t, 2): uple.diff(t)}
        ud_repl  = {ukuw.diff(t): akuw, uple.diff(t): aple}

        logger.debug("Setting and Solving Holonomic Constraints for dependent variables.")

        holonomic_disp = KOL.masscenter.pos_from(N.masscenter).dot(N.frame.x) # = 0
        q_dep_repl  = {qple: sm.solve(holonomic_disp, qple)[0]}

        kol_pos = KOL.masscenter.pos_from(N.masscenter).dot(N.frame.y).xreplace(q_dep_repl)
        kol_pos = kol_pos.xreplace({qkuw: 0., dir: -1, rKUW: 20., lPLE: 71.4})
        if kol_pos < 0.:
            q_dep_repl  = {qple: sm.solve(holonomic_disp, qple)[1]}

        holonomic_velo = holonomic_disp.diff(t).xreplace(qd_repl) # = 0.
        u_dep_repl  = {uple: sm.solve(holonomic_velo, uple)[0].xreplace(q_dep_repl)}

        holonomic_acce = holonomic_velo.diff(t).xreplace(ud_repl).xreplace(qd_repl) # = 0.
        # ud_dep_repl = {uple.diff(t): sm.solve(holonomic_acce, uple.diff(t))[0].xreplace(udep_repl).xreplace(qdep_repl)}
        a_dep_repl = {aple: sm.solve(holonomic_acce, aple)[0].xreplace(u_dep_repl).xreplace(ud_repl).xreplace(q_dep_repl)}

        logger.debug("Lambdifying Parts positions, velocities and accelerations.")
        self.repl        = {2: {ukuw.diff(t): akuw, uple.diff(t): aple},
                            1: {qkuw.diff(t): ukuw, qple.diff(t): uple},
                            0: {},}
        self.dependent   = {2: a_dep_repl,
                            1: u_dep_repl,
                            0: q_dep_repl}

        self.bodies = [self.KUW, self.WNG, self.UNW, self.HZP, self.PLE, self.BLZ, self.KOL]

        self._setup = True

    def lambdify(self):
        logger.info("Lambdifying Positions, Velocities and Accelerations.")
        for body in self.bodies:
            body.lambdify(self.a, self.u, self.q, self.p, self.repl, self.dependent)
        self._lambdified = True

    def set_p(self, setup: dict = {}):
        logger.info("Setting model parametric values.")
        # self.p = [dir, rKUW, eKUW, mKUW, JKUW, rUNW, mUNW, aUNW, lPLE, ePLE, mPLE, JPLE, mKOL, JKOL]
        self.pdict = {self.pnames["dir"]:  setup["motor"]["drehrichtung"],
                      self.pnames["rKUW"]: setup["geometrie"]["kurbelradius"],
                      self.pnames["eKUW"]: setup["geometrie"]["excentrizitaet"],
                      self.pnames["mKUW"]: setup["masse"]["kurbelwelle"],
                      self.pnames["JKUW"]: setup["masse"]["kurbelwelle inertia"],
                      self.pnames["rUNW"]: setup["geometrie"]["kurbelradius"] / 2.,
                      self.pnames["mUNW"]: setup["unwucht"]["masse"] / (setup["geometrie"]["kurbelradius"] / 2),
                      self.pnames["aUNW"]: setup["unwucht"]["winkel"],
                      self.pnames["lPLE"]: setup["geometrie"]["pleuellaenge"],
                      self.pnames["ePLE"]: setup["geometrie"]["hubzapfenlager-pleuel cog"],
                      self.pnames["mPLE"]: setup["masse"]["pleuel"],
                      self.pnames["JPLE"]: setup["masse"]["pleuel inertia"],
                      self.pnames["mKOL"]: setup["masse"]["kolben"],
                      self.pnames["JKOL"]: setup["masse"]["kolben inertia"]}

        self.pvals = [self.pdict[p] for p in self.p]
        for body in self.bodies:
            body.set_p(self.pvals)
        self._p_set = True

    def animate(self, num_rotations: int = 1, speed: float = 13000. / 60. * 2 * np.pi):
        logger.debug(f"Animating Kurbeltrieb results.")

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("Kurbeltrieb")

        gspec = gs.GridSpec(3, 5, wspace=0.5, hspace=0.2)

        ax  = fig.add_subplot(gspec[:,  :2])
        axd = fig.add_subplot(gspec[0, 2:-1])
        axv = axd.twinx()
        axa = axd.twinx()
        axa.spines.right.set_position(("axes", 1.3))

        axf = fig.add_subplot(gspec[1, 2:-1])

        axd.set_ylabel("Kolben Displacement", color="red")
        axv.set_ylabel("Kolben Velocity",     color="blue")
        axa.set_ylabel("Kolben Acceleration", color="orange")

        # axv = fig.add_subplot(gspec[1,3:-1])
        # axa = fig.add_subplot(gspec[2,3:-1])

        for body in self.bodies:
            body.set_plotting(ax)

        self.WNG.set_plotting(ax, d=2.5, plot_props={"color":  "lime"})
        self.UNW.set_plotting(ax, d=2.5, plot_props={"color": "green"})

        qkuw = np.arange(0., num_rotations * 2 * np.pi, np.pi / STEPS_PER_ROTATION, dtype=float)
        ukuw = np.array([speed] * qkuw.shape[0], dtype=float)
        akuw = np.array([0.] * qkuw.shape[0], dtype=float)

        for body in self.bodies:
            body.plot(akuw[0], ukuw[0], qkuw[0])

        qkol = self.KOL.pos(akuw, ukuw, qkuw)
        vkol = self.KOL.vel(akuw, ukuw, qkuw)
        akol = self.KOL.acc(akuw, ukuw, qkuw)

        qhzp = self.HZP.pos(akuw, ukuw, qkuw)

        dky, = axd.plot(qkuw[0], qkol[0, 1], color="red",    label="$d_{y,kolben}$")
        vky, = axv.plot(qkuw[0], vkol[0, 1], color="blue",   label="$v_{y,kolben}$")
        aky, = axa.plot(qkuw[0], akol[0, 1], color="orange", label="$a_{y,kolben}$")

        ax.set_xlim(2.5 * np.min(qhzp[:,0]), 2.5 * np.max(qhzp[:,0]))
        ax.set_ylim(2.5 * np.min(qhzp[:,1]), 1.5 * np.max(qkol[:,1]))
        ax.set_aspect("equal")

        axd.set_xlim(np.min(qkuw), np.max(qkuw))
        axv.set_xlim(np.min(qkuw), np.max(qkuw))
        axa.set_xlim(np.min(qkuw), np.max(qkuw))

        for ax_, v in zip([axd, axv, axa], [qkol[:,1], vkol[:,1], akol[:,1]]):
            n, x = np.min(v), np.max(v)
            dv = np.abs(x - n)
            ax_.set_ylim(n - 0.1 * dv, x + 0.1 * dv)
        axd.legend([dky, vky, aky], [k.get_label() for k in [dky, vky, aky]])

        def animate(frame):
            logger.debug(f"Plotting frame {frame:n}")
            for body in self.bodies:
                body.plot(akuw[frame], ukuw[frame], qkuw[frame])

            dky.set_data(qkuw[:frame+1], qkol[:frame+1,1])
            vky.set_data(qkuw[:frame+1], vkol[:frame+1,1])
            aky.set_data(qkuw[:frame+1], akol[:frame+1,1])

            return dky, vky, aky

        ani = animation.FuncAnimation(fig, animate, interval=100., frames=qkuw.shape[0], repeat=True)
        plt.show(block=True)


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
    _LOG_CH.setLevel(logging.DEBUG)

    kt = Kurbeltrieb()
    kt.lambdify()
    kt.set_p(default_kurbeltrieb)
    kt.animate(3)

