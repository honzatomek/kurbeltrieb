#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as patches
import matplotlib.animation as animation


TIMESTEPS_PER_ROTATION = 51
DIGITS = 5
EPS_TOL = 10. ** (-DIGITS)
PLEUEL_DISCRETISATION = 10


class PltObject:
    def __init__(self, ax, props: dict = {}, **kwargs):
        self.ax = ax
        self.obj = None
        self.props = props
        self.points = np.array([0., 0.])

    def plot(self, coor: np.ndarray, alpha: float):
        T = np.array([[ np.cos(alpha), np.sin(alpha)],
                      [-np.sin(alpha), np.cos(alpha)]], dtype=float)
        if self.obj is None:
            self.obj = plt.Polygon(self.points @ T.T + coor, **self.props)
            self.ax.add_artist(self.obj)
        else:
            self.obj.set_xy(self.points @ T.T + coor)



class Kolben(PltObject):
    def __init__(self, ax, props: dict = {}, width: float = 54.0, height: float = 50.):
        super().__init__(ax, props)
        self.points = np.array([[      -width / 2.,        -height / 2.],
                                [      -width / 2.,         height / 2.],
                                [       width / 2.,         height / 2.],
                                [       width / 2., - 0.5 * height / 2.],
                                [-0.5 * width / 2.,        -height / 2.]], dtype=float)
        self.points += np.array([0., height / 6], dtype=float)



class Wange(PltObject):
    def __init__(self, ax, props: dict = {}, hub: float = 36.96):
        super().__init__(ax, props)

        h = hub
        w = 2 * hub
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



class Pleuel(PltObject):
    def __init__(self, ax, props: dict = {}, length: float = 36.96, width: float = 8.):
        super().__init__(ax, props)
        points = []

        num = 6
        da = (np.pi /  2) / (num - 1)
        a = np.pi - da
        for i in range(num):
            a += da
            points.append([width / 2 * np.cos(a), width / 2 * np.sin(a)])

        for point in reversed(points):
            points.append([length - point[0], point[1]])

        for point in reversed(points[1:-1]):
            points.append([point[0], -point[1]])

        self.points = np.array(points, dtype=float)



class Pin(PltObject):
    def __init__(self, ax, props: dict = {}, diameter: float = 10.):
        super().__init__(ax, props)
        points = []

        num = 36
        da = 2 * np.pi / (num - 1)
        a = 0. - da
        for i in range(num):
            a += da
            points.append([diameter / 2 * np.cos(a), diameter / 2 * np.sin(a)])

        self.points = np.array(points, dtype=float)



class Vector(PltObject):
    def __init__(self, ax, props: dict = {}):
        super().__init__(ax, props)

    def plot(self, coor: np.ndarray, F: np.ndarray, s: float = 1.0):
        if self.obj is None:
            self.obj = self.ax.arrow(coor[0], coor[1], F[0] * s, F[1] * s, **self.props)
        else:
            self.obj.set_data(x=coor[0], y=coor[1], dx=F[0] * s, dy=F[1] * s)



class Kurbeltrieb:
    def __init__(self, hub: float = 10., pleuel: float = 70., rpm: float = 9300., m_kolben: float = 112.E-6, unwucht: tuple = (0.002267815, 4.974311637), m_pleuel: float = 58.7E-6):
        """Kurbeltrieb setup - use consistent units (e.g. mm/s/rad/tonnes etc.)

        Args:
            hub (float):       the cranshaft stroke length
            pleuel (float):    the conrod length
            rpm (float):       the motor Rotations per Minute
            m_kolben (float):  the piston mass
            unwucht (tuple):   (the imbalance in mass * length, the angle of the imbalance from y-axis when piston is in OT)
        """
        self.h = hub
        self.p = pleuel
        self.alpha0 = np.pi / 2.            # OT position
        self.omega = 2 * np.pi * rpm / 60.  # rad / s
        self.m_k = m_kolben # t
        self.m_p = m_pleuel # t
        self.u_w = unwucht  # (Unwucht Wange + Schwungrad t * mm, Unwucht Winkel von x-Achse wann Kolben im OT [rad])

        self.t = [0.]  # initial time for animation


    def alpha(self, t):
        """KUW angular position from OT as a function of time."""
        # if isinstance(t, float):
        #     if t < 0.:
        #         raise ValueError('Each time t must be greater than 0')
        # elif not all(t >= 0.):
        #     raise ValueError('Each time t must be greater than 0')
        alpha = self.omega * t
        return np.array([alpha], dtype=float).flatten()

    def hubzapfen_xy(self, t):
        """kolben - pleuel x, y position as a function of time."""
        h_x = (self.h / 2) * np.sin(self.alpha(t))
        h_y = (self.h / 2) * np.cos(self.alpha(t))
        return np.array([h_x, h_y], dtype=float)

    def alpha_g(self, t):
        """KUW global angle in GCS"""
        h_x, h_y = self.hubzapfen_xy(t)
        alpha = np.arctan2(h_y, h_x)
        return np.array([alpha], dtype=float).flatten()

    def kolben_xy(self, t):
        """kolben x, y position as a function of time"""
        A = self.alpha(t)
        if isinstance(t, float):
            k_x = np.array([0.], dtype=float)
        else:
            k_x = np.zeros(t.shape)

        sqr = (self.p ** 2 - (self.h / 2) ** 2 * (np.sin(A) ** 2)) ** 0.5
        k_y = (self.h / 2) * np.cos(A) + sqr

        return np.array([k_x, k_y], dtype=float)

    def beta(self, t):
        """angular position of pleuel (angle from y axis)  as a function of time."""
        h_x, h_y = self.hubzapfen_xy(t)
        k_x, k_y = self.kolben_xy(t)
        dx = k_x - h_x
        dy = k_y - h_y
        beta = np.arctan2(k_y - h_y, k_x - h_x) - np.pi / 2
        return np.array([beta], dtype=float).flatten()

    def kolben_vxy(self, t):
        """kolben speed as a function of time"""
        A = self.alpha(t)
        dt = (1 / self.omega) / 1000.
        if isinstance(t, float):
            v_x = np.array([0.], dtype=float)
            t = np.array([t, t + dt], dtype=float)
        else:
            v_x = np.zeros(t.shape)
            t = np.hstack([t, [t[-1] + dt]])
        k_y = np.array([yy for xx, yy in [self.kolben_xy(tt) for tt in t]], dtype=float).flatten()
        v_y = (k_y[1:] - k_y[:-1]) / (t[1:] - t[:-1])
        return np.array([v_x, v_y], dtype=float)

    def kolben_axy(self, t):
        """kolben acceleration as a function of time"""
        A = self.alpha(t)
        dt = (1 / self.omega) / 1000.
        if isinstance(t, float):
            a_x = np.array([0.], dtype=float)
            t = np.array([t, t + dt], dtype=float)
        else:
            a_x = np.zeros(t.shape)
            t = np.hstack([t, [t[-1] + dt]])

        v_y = np.array([yy for xx, yy in [self.kolben_vxy(tt) for tt in t]], dtype=float).flatten()
        a_y = (v_y[1:] - v_y[:-1]) / (t[1:] - t[:-1])

        a = self.h / 2
        r = self.p

        gamma = a / r
        a_confluence = a * (self.omega ** 2) * (np.cos(A) + gamma * np.cos(2 * A))
        # return a_x, a_y, a_confluence
        return np.array([a_x, a_y], dtype=float)

    def pleuel_xy(self, t, position: float = 0.5):
        """position of a point on pleuel in time"""
        hx, hy = self.hubzapfen_xy(t)
        kx, ky = self.kolben_xy(t)

        return np.array([hx + (kx - hx) * position, hy + (ky - hy) * position], dtype=float)


    def kolben_Fxy(self, t):
        """forces from kolben motion acting on the kolben"""
        # A = self.alpha(t)
        k_x, k_y = self.kolben_xy(t)
        a_x, a_y = self.kolben_axy(t)
        beta = self.beta(t)

        F_y = - a_y * self.m_k
        F_x = - F_y * np.tan(beta)
        M_z = F_x * k_x

        return np.array([F_x, F_y, M_z], dtype=float)

    def wange_Fxy(self, t):
        """forces from wange motion acting on the KUW bearings"""
        gamma = self.alpha_g(t) + self.u_w[1]

        W_r = self.u_w[0] * self.omega ** 2

        W_x = W_r * np.cos(gamma)
        W_y = W_r * np.sin(gamma)

        return np.array([W_x, W_y], dtype=float)

    def pleuel_Fxy(self, t):
        """forces from pleuel movement acting on KUW bearings"""
        dt = (1 / self.omega) / 1000.

        numseg = PLEUEL_DISCRETISATION
        axy = []
        # pos = []
        radius = []
        for i in range(numseg + 1):
            x1, y1 = self.pleuel_xy(t - dt, position=i / numseg)
            x2, y2 = self.pleuel_xy(t     , position=i / numseg)
            x3, y3 = self.pleuel_xy(t + dt, position=i / numseg)
            axy.append([((x3 - x2) / dt - (x2 - x1) / dt) / dt,
                        ((y3 - y2) / dt - (y2 - y1) / dt) / dt])
            # pos.append([x2, y2])
            radius.append(self.p * i / numseg)
        axy = np.array(axy, dtype=float)
        # pos = np.array(pos, dtype=float)
        radius = np.array(radius, dtype=float)

        Pxy = -axy * self.m_p / numseg

        hx, hy = self.hubzapfen_xy(t)
        kx, ky = self.kolben_xy(t)

        # direction of pleuel in time
        x = np.array([kx - hx, ky - hy], dtype=float).T
        x /= np.linalg.norm(x)
        x = np.hstack([x, np.zeros([x.shape[0], 1], dtype=float)])
        y = -np.cross(np.array([0., 0., 1.], dtype=float), x)[:,:2]
        y /= np.linalg.norm(y)

        # perpendicular projection of forces on pleuel y axis
        Pperp = []
        for f in Pxy:
            Pperp.append(np.sum(f.T * y, axis=1))
        Pperp = np.array(Pperp, dtype=float).round(DIGITS)

        # perpendicular projection of sum on pleuel y axis
        Psum = np.sum(Pxy, axis=0).T.round(DIGITS)
        Psum_proj = np.sum(Psum * y, axis=1).round(DIGITS)

        # moment of forces
        Mperp = np.sum(Pperp * radius.reshape(numseg+1,1), axis=0).round(DIGITS)
        Mperp[np.abs(Mperp) < EPS_TOL] = 0.

        r = Mperp / Psum_proj
        for i in range(r.shape[0]):
            if np.isnan(r[i]):
                if i == 0:
                    r[i] = r[i+1]
                else:
                    r[i] = r[i-1]

        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.plot(radius, Pperp[:,10])
        # plt.show()

        xy = self.pleuel_xy(t     , position=r / self.p)

        return Psum.T, xy

    def lager_Fxy(self, t):
        """forces from wange and kolben acting on the KUW bearings"""
        Fx, Fy, _ = self.kolben_Fxy(t)
        Wx, Wy    = self.wange_Fxy(t)
        Pxy, Ppos = self.pleuel_Fxy(t)

        Px, Py = np.sum(Pxy, axis=0)

        return np.array([Wx + Fx + Px, Wy + Fy + Py], dtype=float)

    def animate_kurbeltrieb(self, num_rotations: int = 2, filename: str = None):
        end_time = int(num_rotations) * ((2 * np.pi) / self.omega)

        dt = (2 * np.pi) / self.omega / TIMESTEPS_PER_ROTATION

        self.t = np.linspace(0., end_time, int(end_time / dt) + 1)

        alpha     = self.alpha(self.t)
        alpha_g   = self.alpha_g(self.t)
        hx, hy    = self.hubzapfen_xy(self.t)
        kx, ky    = self.kolben_xy(self.t)
        vx, vy    = self.kolben_vxy(self.t)
        ax, ay    = self.kolben_axy(self.t)
        Fx, Fy, _ = self.kolben_Fxy(self.t)
        Wx, Wy    = self.wange_Fxy(self.t)

        Pxy, Ppos = self.pleuel_Fxy(self.t)

        Px, Py = Pxy

        Rx = Fx + Wx + Px
        Ry = Fy + Wy + Py

        kraft_scale = self.p / (2 * (Wx[0] ** 2. + Wy[0] ** 2) ** 0.5)


        fig = plt.figure(figsize=(18, 10))
        gspec = gs.GridSpec(3, 5)

        ax_2d = fig.add_subplot(gspec[:, :2])
        ax_k  = fig.add_subplot(gspec[0, 2:-1])
        # ax_v  = fig.add_subplot(gspec[1, 2:])
        # ax_a  = fig.add_subplot(gspec[2, 2:])
        ax_v = ax_k.twinx()
        ax_a = ax_k.twinx()
        ax_a.spines.right.set_position(("axes", 1.2))
        ax_f  = fig.add_subplot(gspec[1, 2:-1])
        ax_r  = fig.add_subplot(gspec[2, 2:-1])

        ax_2d.set_xlim([-1.5 * self.h, 1.5 * self.h])
        ax_2d.set_ylim([-2* self.h, 2 * self.h + self.p])
        ax_2d.set_aspect("equal")


        wange = Wange(ax_2d, {"closed":       True,
                              "fill":         True,
                              "edgecolor": "black",
                              "facecolor": "black",
                              "zorder":          0})

        kolben = Kolben(ax_2d, {"closed":       True,
                                "fill":         True,
                                "edgecolor": "black",
                                "facecolor": "black",
                                "zorder":         25})

        pleuel = Pleuel(ax_2d, {"closed":       True,
                                "fill":         True,
                                "edgecolor":  "grey",
                                "facecolor":  "grey",
                                "zorder":         50}, length=self.p)

        bolzen = Pin(ax_2d, {"closed":       True,
                              "fill":         True,
                              "edgecolor": "white",
                              "facecolor": "white",
                              "zorder":         26}, diameter=12.)

        hubzapfen = Pin(ax_2d, {"closed":       True,
                                "fill":         True,
                                "edgecolor": "white",
                                "facecolor": "white",
                                "zorder":          5}, diameter=12.)

        kkraft = Vector(ax_2d, {"color": "red",
                                "shape": "full",
                                "length_includes_head": True,
                                "head_width": 5.,
                                "zorder": 100})

        wkraft = Vector(ax_2d, {"color": "blue",
                                "shape": "full",
                                "length_includes_head": True,
                                "head_width": 5.,
                                "zorder":  1})

        pkraft = Vector(ax_2d, {"color": "orange",
                                "shape": "full",
                                "length_includes_head": True,
                                "head_width": 5.,
                                "zorder": 51})

        kolben.plot(np.array([kx[0], ky[0]], dtype=float), alpha=0.)
        wange.plot(np.array([0., 0.], dtype=float), alpha=alpha[0])
        pleuel.plot(np.array([hx[0], hy[0]], dtype=float), alpha=-np.arctan2(ky[0] - hy[0], kx[0] - hx[0]))
        bolzen.plot(np.array([kx[0], ky[0]], dtype=float), alpha=0.)
        hubzapfen.plot(np.array([hx[0], hy[0]], dtype=float), alpha=0.)
        kkraft.plot(np.array([kx[0], ky[0]], dtype=float),
                    np.array([Fx[0], Fy[0]], dtype=float), kraft_scale)
        wkraft.plot(np.array([   0.,    0.], dtype=float),
                    np.array([Wx[0], Wy[0]], dtype=float), kraft_scale)

        pkraft.plot(Ppos[:,0], Pxy[:,0], kraft_scale)


        kolben_xy, = ax_k.plot(self.t[0], ky[0], color="red", label="k_y")
        kolben_vy, = ax_v.plot(self.t[0], vy[0], color="blue", label="v_y")
        kolben_ay, = ax_a.plot(self.t[0], ay[0], color="orange", label="a_y")

        kolben_Fx, = ax_f.plot(self.t[0], Fx[0], color="red", linestyle="--", label="K_x")
        kolben_Fy, = ax_f.plot(self.t[0], Fy[0], color="red", label="K_y")
        wange_Fx,  = ax_f.plot(self.t[0], Wx[0], color="blue", linestyle="--", label="W_x")
        wange_Fy,  = ax_f.plot(self.t[0], Wy[0], color="blue", label="W_y")
        pleuel_Fx, = ax_f.plot(self.t[0], Px[0], color="orange", linestyle="--", label="P_x")
        pleuel_Fy, = ax_f.plot(self.t[0], Py[0], color="orange", label="P_y")

        kuw_Rx, = ax_r.plot(self.t[0], Rx[0], label="R_x")
        kuw_Ry, = ax_r.plot(self.t[0], Ry[0], label="R_y")

        ax_v.plot([self.t[0], self.t[-1]], [0., 0.], linewidth=1, color="black", zorder=-1)
        ax_f.plot([self.t[0], self.t[-1]], [0., 0.], linewidth=1, color="black", zorder=-1)
        ax_r.plot([self.t[0], self.t[-1]], [0., 0.], linewidth=1, color="black", zorder=-1)

        for ax, v in zip([ax_k, ax_v, ax_a, ax_f, ax_r],
             [[ky], [vy], [ay], [Fx, Fy, Wx, Wy, Rx, Ry], [Fx, Fy, Wx, Wy, Rx, Ry]]):
            ax.set_xlim([0., self.t[-1]])
            minv, maxv = np.min(v), np.max(v)
            ax.set_ylim([minv - 0.1 * np.abs(maxv - minv), maxv + 0.1 * np.abs(maxv - minv)])
            # ax.set_ylabel(lab)
            # ax.legend()
        ax_k.set_ylabel("Kolben Displacement", color="red")
        ax_v.set_ylabel("Kolben Velocity", color="blue")
        ax_a.set_ylabel("Kolben Acceleration", color="orange")
        ax_f.set_ylabel("Inertia Forces on KUW")
        ax_r.set_ylabel("Resulting Forces on KUW")
        lns = [kolben_xy, kolben_vy, kolben_ay]
        lab = [l.get_label() for l in lns]
        ax_a.legend(lns, lab, loc="best")
        ax_f.legend()
        ax_r.legend()

        fig.suptitle("Kurbeltrieb Inertia Forces decomposition using Numerical Analysis")


        def animate(frame: int):
            kolben.plot(np.array([kx[frame], ky[frame]], dtype=float), alpha=0.)
            wange.plot(np.array([0., 0.], dtype=float), alpha=alpha[frame])
            pleuel.plot(np.array([hx[frame], hy[frame]], dtype=float),
                        alpha=-np.arctan2(ky[frame] - hy[frame], kx[frame] - hx[frame]))
            bolzen.plot(np.array([kx[frame], ky[frame]], dtype=float), alpha=0.)
            hubzapfen.plot(np.array([hx[frame], hy[frame]], dtype=float), alpha=0.)
            kkraft.plot(np.array([kx[frame], ky[frame]], dtype=float),
                        np.array([Fx[frame], Fy[frame]], dtype=float), kraft_scale)
            wkraft.plot(np.array([       0.,        0.], dtype=float),
                        np.array([Wx[frame], Wy[frame]], dtype=float), kraft_scale)
            pkraft.plot(Ppos[:,frame], Pxy[:,frame], kraft_scale)

            for plot, vals in zip([kolben_xy, kolben_vy, kolben_ay, kolben_Fx, kolben_Fy,
                                   wange_Fx, wange_Fy, pleuel_Fx, pleuel_Fy, kuw_Rx, kuw_Ry],
                                  [ky, vy, ay, Fx, Fy, Wx, Wy, Px, Py, Rx, Ry]):
                plot.set_data(self.t[:frame+1], vals[:frame+1])

            return kolben, wange, pleuel, hubzapfen, kkraft, wkraft

        ani = animation.FuncAnimation(fig, animate, interval=100., frames=len(self.t), repeat=True)

        if filename is not None:
            print(f"[+] Saving animation to: {filename:s}.")
            writergif = animation.PillowWriter(fps=30)
            ani.save(filename, writer=writergif)

        plt.show(block=True)

    def plot_forces(self):
        end_time = ((2 * np.pi) / self.omega)

        dt = (2 * np.pi) / self.omega / TIMESTEPS_PER_ROTATION
        self.t = np.linspace(0., end_time, int(end_time / dt) + 1)

        Fx, Fy, _ = self.kolben_Fxy(self.t)
        Wx, Wy    = self.wange_Fxy(self.t)
        Pxy, Ppos = self.pleuel_Fxy(self.t)
        Px, Py = Pxy
        Rx = Fx + Wx + Px
        Ry = Fy + Wy + Py

        fig = plt.figure()
        ax = fig.add_subplot()

        R = (Rx ** 2 + Ry ** 2) ** 0.5
        Rmax = np.max(R)

        arrowdict = {"shape": "full",
                     "length_includes_head": True,
                     "head_width": Rmax / 20}

        kolben, = ax.plot(Fx, Fy, color="red", label="Fkolben")
        ak      = ax.arrow(Fx[0], Fy[0], Fx[1] - Fx[0], Fy[1] - Fy[0], color="red", **arrowdict)


        wange,  = ax.plot(Wx, Wy, color="blue", label="Fwange")
        wk      = ax.arrow(Wx[0], Wy[0], Wx[1] - Wx[0], Wy[1] - Wy[0], color="blue", **arrowdict)

        pleuel, = ax.plot(Px, Py, color="orange", label="Fpleuel")
        pk      = ax.arrow(Px[0], Py[0], Px[1] - Px[0], Py[1] - Py[0], color="orange", **arrowdict)

        kuw,    = ax.plot(Rx, Ry, color="black", label="ΣF")
        kk      = ax.arrow(Rx[0], Ry[0], Rx[1] - Rx[0], Ry[1] - Ry[0], color="black", **arrowdict)

        ax.legend()
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")

        plt.show()




if __name__ == "__main__":
    name = "MB03_kurbeltrieb.gif"
    name = None
    kuw_r                 =    18.48     # mm
    conrod                =    65.5      # mm
    rpm                   = 13000.       # rot / min
    conrod_mass           =    58.7E-6   # t
    kolben_mass           =   112.0E-6   # t - inluding Bolzen
    kuw_mass              =     1.105E-3 # t - inluding Hubzapfenlager
    kuw_excentricity      =     2.05137  # mm
    kuw_angle_COG_from_OT = np.pi        # rad

    unwucht = (kuw_mass * kuw_excentricity, kuw_angle_COG_from_OT)

    kt = Kurbeltrieb(hub      = 2 * kuw_r,
                     pleuel   = conrod,
                     rpm      = rpm,
                     m_kolben = kolben_mass,
                     unwucht  = unwucht,
                     m_pleuel = conrod_mass)

    kt.animate_kurbeltrieb(num_rotations = 3, filename = name)
    kt.plot_forces()

