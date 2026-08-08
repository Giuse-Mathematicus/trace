#!/usr/bin/env python

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9
VEC_LEN = 42.0

# ============================================================
# Reference frame builder
# ============================================================
# Where Initial Reference frame = Tetrahedron, with 4
# Reference Planes.

def build_reference_frame(
        d01, d12, d20, d03, d13, d23, handedness="right"
        ):
    """
        Build a coordinate reference tetrahedron from six
        positive reference distances. The six reference
        distances are a subset of the measured distances dV
        in the data.

        Statement:
        A left-handed Trace must be represented by a
        left-handed Curve, and a right-handed Trace by a
        right-handed Curve. The six distances determine the
        tetrahedron only up to mirror symmetry.
        Therefore the user must choose the handedness of
        the reference tetrahedron, equal to that of the
        Trace lower strip.

        Convention used here:
        q0 = (0, 0, 0)
        q1 lies on the positive x-axis
        q2 lies in the xy-plane, with y2 > 0
        Then:
        handedness = "right" gives z3 > 0
        handedness = "left" gives z3 < 0
    """
    handedness = handedness.lower().strip()
    if handedness not in ("right", "left", "r", "l"):
        raise ValueError(
            "handedness must be 'right', 'left', 'r', or 'l'"
        )
    q0 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([d01, 0.0, 0.0])
    x2 = (d01**2 + d20**2 - d12**2) / (2 * d01)
    y2_sq = d20**2 - x2**2
    if y2_sq < -1e-6:
        raise ValueError(
            f"Inconsistent reference triangle: "
            f"y2^2={y2_sq}"
        )
    y2 = np.sqrt(max(y2_sq, 0.0))
    if abs(y2) < 1e-9:
        raise ValueError(
            f"Degenerate reference"
            f"triangle: q0, q1, q2 are collinear"
        )
    q2 = np.array([x2, y2, 0.0])
    x3 = (d01**2 + d03**2 - d13**2) / (2 * d01)
    C = (x2**2 + y2**2 + d03**2 - d23**2) / 2
    y3 = (C - x2 * x3) / y2
    z3_sq = d03**2 - x3**2 - y3**2
    if z3_sq < -1e-6:
        raise ValueError(
            f"Inconsistent tetrahedron: z3^2={z3_sq}"
        )
    z3_abs = np.sqrt(max(z3_sq, 0.0))
    if handedness in ("right", "r"):
        z3 = +z3_abs
    else:
        z3 = -z3_abs
    q3 = np.array([x3, y3, z3])
    return q0, q1, q2, q3

# ============================================================
# Trilateration (3 spheres)
# ============================================================
# Assuming 3 {Reference Marks} = qs within {q0, q1, q2, q3}
# of CPlanes.py, and 3 {Reference Distances} = ds of the
# measured Mark from the 3 { Reference Marks }.

def trilaterate_3(qs, ds):

    q1, q2, q3 = qs
    r1, r2, r3 = ds

    ex = (q2 - q1)
    d = np.linalg.norm(ex)
    if d < 1e-9:
        return []

    ex /= d

    i = np.dot(ex, q3 - q1)
    ey = q3 - q1 - i * ex
    j = np.linalg.norm(ey)
    if j < 1e-9:
        return []

    ey /= j
    ez = np.cross(ex, ey)

    x = (r1**2 - r2**2 + d**2) / (2*d)
    y = (r1**2 - r3**2 + i**2 + j**2 - 2*i*x) / (2*j)

    z2 = r1**2 - x**2 - y**2

    if z2 < -1e-6:
        return []

    z = np.sqrt(max(z2, 0.0))

    # Two solutions, of which one is a Mirror Image:

    p1 = q1 + x*ex + y*ey + z*ez
    p2 = q1 + x*ex + y*ey - z*ez

    return [p1, p2]

# ============================================================
# Circle and arc helpers
# ============================================================

def circle_from_triad(p1, p2, p3, eps=1e-9):

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    a = p2 - p1
    b = p3 - p1
    n = np.cross(a, b)

    n2 = np.dot(n, n)

    if n2 < eps:
        return None, np.inf, 0.0

    a2 = np.dot(a, a)
    b2 = np.dot(b, b)

    center = p1 + (
        np.cross(n, a) * b2 +
        np.cross(b, n) * a2
    ) / (2.0 * n2)

    radius = np.linalg.norm(p1 - center)
    curvature = 1.0 / radius

    # A number of 10 extra points in the arc is assumed.
    return center, radius, curvature


def sample_arc_points(p1, p2, p3, n_extra=10, eps=1e-9):

    center, radius, curvature = circle_from_triad(p1, p2, p3)

    if center is None or not np.isfinite(radius):
        return []

    v1 = p1 - center
    v2 = p2 - center
    v3 = p3 - center

    e1 = v1 / np.linalg.norm(v1)

    normal = np.cross(v1, v3)
    normal_norm = np.linalg.norm(normal)

    if normal_norm < eps:
        return []

    normal = normal / normal_norm
    e2 = np.cross(normal, e1)

    def angle_of(v):
        x = np.dot(v, e1)
        y = np.dot(v, e2)
        return np.arctan2(y, x)

    a1 = angle_of(v1)
    a2 = angle_of(v2)
    a3 = angle_of(v3)

    def positive_angle(a_from, a_to):
        d = a_to - a_from
        while d < 0:
            d += 2.0 * np.pi
        return d

    d13 = positive_angle(a1, a3)
    d12 = positive_angle(a1, a2)

    # If p2 is not between p1 and p3 in positive direction,
    # use the opposite orientation.
    if d12 > d13:
        if d13 > 0:
            d13 = d13 - 2.0 * np.pi

    angles = np.linspace(a1, a1 + d13, n_extra + 2)[1:-1]

    extra = []

    for a in angles:
        p = center + radius * (np.cos(a) * e1 + np.sin(a) * e2)
        extra.append(p)

    return extra


# ============================================================
# Exact Frenet-Serret Frames
# for an arc, not for an Even Mark.
# ============================================================
#
# For each Arc A_i generated by the odd-centered Triad:
#
#     { p_(i-1), p_i, p_(i+1) }
#
# the exact Frenet-Serret frame is defined from:
#
#     Curvature Center C_i
#
# and the Radius Vectors:
#
#     V1 = p_(i-1) - C_i
#     V2 = p_(i+1) - C_i
#
# yielding:
#
#     N1 = V1 / ||V1||
#     N2 = V2 / ||V2||
#
#     N1, T1 for p_i-1; N2, T2 for p_i+1; B for p_i-1,p_i+1
#
#     B = (N2 x N1) / ||N2 x N1||
#
#     T1 = N1 x B
#     T2 = N2 x B
#
# Collinear case:
#
#     T = (p_(i+1)-p_(i-1)) / ||p_(i+1)-p_(i-1)||
#
#     N and B are undefined.
#
# These frames are not used here for torsion classification.
# They are used only to build the three comparison plots:
#
#     Plot_T.png
#     Plot_N.png
#     Plot_B.png
#
# Each plot keeps:
#
#     same green Curve background
#     same metric scale
#     same vector length
#
# ============================================================

def normalize(v, eps=1e-9):

    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < eps:
        return None

    return v / n


def exact_fs_for_triad_endpoints(
        p_prev, p_mid, p_next, center, eps=1e-9
    ):
    """
    IMPORTANT ADVISORY.

    To get {T, N, B}, it is necessary to partition the Trace
    into Arcs having:

        0 < Angle( Ni, C_i, N1 ) < Pi

    Angles near Pi are allowed.

    Here:

        N1 = p_(i-1) - C_i
        Ni = p_i     - C_i
        N2 = p_(i+1) - C_i

    The Binormal is computed from:

        B = (Ni x N1) / ||Ni x N1||

    This avoids the orientation ambiguity produced by using
    N2 x N1 when the represented Arc angle is greater than Pi.
    """

    if center is None:
        T = normalize(p_next - p_prev, eps=eps)
        if T is None:
            return None

        return {
            "left":  {"T": T, "N": None, "B": None},
            "middle": {"T": T, "N": None, "B": None},
            "right": {"T": T, "N": None, "B": None},
        }

    V1 = p_prev - center
    Vi = p_mid  - center
    V2 = p_next - center

    N1 = normalize(V1, eps=eps)
    Ni = normalize(Vi, eps=eps)
    N2 = normalize(V2, eps=eps)

    if N1 is None or Ni is None or N2 is None:
        return None

    cross = np.cross(Ni, N1)
    nc = np.linalg.norm(cross)

    if nc < eps:
        T = normalize(p_next - p_prev, eps=eps)
        if T is None:
            return None

        return {
            "left":  {"T": T, "N": None, "B": None},
            "middle": {"T": T, "N": None, "B": None},
            "right": {"T": T, "N": None, "B": None},
        }

    B = cross / nc

    T1 = normalize(np.cross(N1, B), eps=eps)
    Ti = normalize(np.cross(Ni, B), eps=eps)
    T2 = normalize(np.cross(N2, B), eps=eps)

    return {
        "left":   {"T": T1, "N": N1, "B": B},
        "middle": {"T": Ti, "N": Ni, "B": B},
        "right":  {"T": T2, "N": N2, "B": B},
    }

# ============================================================
# Main class
# ============================================================

class CPlanes:

    def __init__(self, ref_distances, dv, tol=1.0,
                 handedness='right', curve_name=None):

        self.qs = np.array(build_reference_frame(
            *ref_distances, handedness
        ))
        self.dv = dv
        self.tol = tol

        self.points = {}
        self.errors = {}

        self.triads = {}
        self.arcs = {}
        self.curvatures = {}
        self.extra_points = {}

        # Layer 2: exact Frenet-Serret frame data.
        self.arc_frames = {}
        self.TPlot = {}
        self.NPlot = {}
        self.BPlot = {}

        self.curve_name = curve_name

    # --------------------------------------------------------

    def reconstruct(self):

        # For all Marks of the Matrix
        #   { Marks: Mark Distance Vectors } = dv
        # Reconstruction (4-plane method) applies
        # Trilateration 4 times to each Mark {m} of dv, each
        # Trilateration, is based on one plane.

        print("\n--- Reconstruction (4-plane method) ---")

        # The 4 planes of the Tetrahedron:
        #   {q0, q1, q2}, {q0, q1, q3},
        #   {q1, q2, q3}, {q2, q0, q3},
        # are shortly written here:

        planes = [
            (0,1,2),
            (0,1,3),
            (1,2,3),
            (2,0,3)
        ]

        # For a Mark m and d an element of Reference Marks:
        # Both m, d in the Matrix
        #   { Marks; Marks'Distance Vectors } = dv.

        for m, d in self.dv.items():

            candidates = []

            for (i,j,k) in planes:

                qs = [self.qs[i], self.qs[j], self.qs[k]]
                ds = [d[i], d[j], d[k]]

                # sols are output from one plane.                 
                # For each Mark {m}, a Trilateration gives 0,
                # 1 or 2 Solutions, (one is a mirror image).
                # For each Mark {m}, the 4 Trilaterations give
                # 8 Solution candidates.
                # For each Mark {m}, the 8 Solutions are
                # stored in its array {candidates}.
                # That is, each Mark {m} has its array
                # {candidates}.    
                sols = trilaterate_3(qs, ds)
                candidates.extend(sols)

            if not candidates:
                print(f"Mark {m}: NO SOLUTION")
                self.points[m] = None
                self.errors[m] = None
                continue

            # The Reconstruction is applied to all Marks
            # listed in dv. 
            # Continuing "Reconstruction":
            # ------------------------------------------------
            # Cluster candidates
            # ------------------------------------------------
            # For each Mark {m}, a selection must be performed
            # in its array named {cluster candidates} of {m},
            # introducing the array {accepted}.
            # The Mark m is considered, but also all the Marks
            # p, q within {candidates} of {m} are considered.
            # The positions of p, q are compared; p is the
            # judged candidate. 

            accepted = []

            for p in candidates:
                good = True
                for q in candidates:
                    if np.linalg.norm(p - q) > self.tol:
                        good = False
                        break
                if good:
                    accepted.append(p)

            if not accepted:
                accepted = candidates

            # In the array {accepted} of p:    
            # one reconstructed Mark p must be defined,
            # making a mean: it is named p_final and it will
            # become the reconstructed m.

            p_final = np.mean(accepted, axis=0)

            err = np.mean([
                abs(np.linalg.norm(p_final - self.qs[i]) - d[i])
                for i in range(4)
            ])

            # Then, for the Mark m:
            # the p_final becomes the reconstructed m, etc.:
            self.points[m] = p_final
            self.errors[m] = err

            print(
                f"Mark {m:2d}: candidates={len(candidates)}"
                f" err={err:.3f}"
            )

            # As it is said at the beginning of Reconstruct,
            # the Reconstruction is done for all Marks listed
            # in dv.

        self.build_triads_arcs_and_extra_points()
        self.build_frenet_frame_comparisons()

    # --------------------------------------------------------

    def build_triads_arcs_and_extra_points(self, n_extra=10):

        print("\n--- Triads, Arcs, Curvatures ---")

        self.triads = {}
        self.arcs = {}
        self.curvatures = {}
        self.extra_points = {}
        self.arc_frames = {}

        marks = sorted([
            m for m, p in self.points.items()
            if p is not None
        ])

        # Odd-centered triads:
        # {p_(i-1), p_i, p_(i+1)} for odd i
        for i in marks:

            if i % 2 == 0:
                continue

            if (i - 1) not in self.points:
                continue
            if (i + 1) not in self.points:
                continue
            if self.points[i - 1] is None:
                continue
            if self.points[i] is None:
                continue
            if self.points[i + 1] is None:
                continue

            p_prev = self.points[i - 1]
            p_mid  = self.points[i]
            p_next = self.points[i + 1]

            self.triads[i] = (i - 1, i, i + 1)

            center, radius, curvature = circle_from_triad(
                p_prev, p_mid, p_next
            )

            self.arcs[i] = {
                "marks": (i - 1, i, i + 1),
                "center": center,
                "radius": radius
            }

            self.curvatures[i] = curvature

            self.extra_points[i] = sample_arc_points(
                p_prev, p_mid, p_next,
                n_extra=n_extra
            )

            self.arc_frames[i] = exact_fs_for_triad_endpoints(
                p_prev, p_mid, p_next, center
            )

            print(
                f"A_{i}: triad=({i-1},{i},{i+1}) "
                f"radius={radius:.3f}  k={curvature:.6f}"
            )

    # --------------------------------------------------------

    def build_frenet_frame_comparisons(self):

        # For each Even Mark e, one can compare:
        #
        # incoming frame from Arc A_(e-1), at its right
        # endpoint,
        # outgoing frame from Arc A_(e+1), at its left
        # endpoint.
        #
        # This is exactly the comparison of Curve parts at
        # the same Mark. It is not a torsion classification;
        # it is only stored for plotting.

        print("\n--- Frenet-Serret frame comparisons ---")

        self.TPlot = {}
        self.NPlot = {}
        self.BPlot = {}

        marks = sorted([
            m for m, p in self.points.items()
            if p is not None
        ])

        for e in marks:

            # We skip odd marks, so we will compare only even
            # ones. 
            if e % 2 != 0:
                continue

            left_arc = e - 1
            right_arc = e + 1

            if left_arc not in self.arc_frames:
                continue
            if right_arc not in self.arc_frames:
                continue

            fs_left = self.arc_frames[left_arc]
            fs_right = self.arc_frames[right_arc]

            if fs_left is None or fs_right is None:
                continue

            # ------------------------------------------------
            # Advisory: Cusps.
            #
            # It is advisable to represent a Cusp by its
            # incoming {T,N,B} and its outgoing {T,N,B} at its
            # vertex.
            #
            # Therefore, in this program, the Cusp vertex
            # should be an Even Mark.
            #
            # The previous Triad gives the incoming frame at
            # the Cusp. The next Triad gives the outgoing
            # frame at the Cusp.
            #
            # At a true Cusp, the inversion of T is mandatory
            # and must not be automatically corrected,
            # because it represents the abrupt inversion of
            # the path while the indexed Curve remains
            # continuous.
            #
            # Superposed or nearly superposed Marks are
            # allowed: each Mark keeps its own Index value,
            # and the Index is the directed quantized
            # parameter of the Curve.
            # ------------------------------------------------
            incoming = fs_left["right"]
            outgoing = fs_right["left"]

            T0 = incoming["T"]
            T1 = outgoing["T"]
            N0 = incoming["N"]
            N1 = outgoing["N"]
            B0 = incoming["B"]
            B1 = outgoing["B"]

            # ------------------------------------------------
            # Orientation correction.
            #
            # Main goal:
            # avoid artificial inversions of T along the Curve
            # parameter t.
            #
            # If incoming and outgoing T vectors point in
            # opposite directions, and no cusp is
            # intentionally being represented, reverse the
            # outgoing Frenet frame.
            # ------------------------------------------------

            #if T0 is not None and T1 is not None:
            #    if np.dot(T0, T1) < 0.0:
            #        T1 = -T1
            #        if N1 is not None:
            #            N1 = -N1
            #        if B1 is not None:
            #            B1 = -B1
            
            # B = (Ni x N1) / ||Ni x N1||

            if T0 is not None and T1 is not None:
                self.TPlot[e] = (T0, T1)

            if N0 is not None and N1 is not None:
                self.NPlot[e] = (N0, N1)

            if B0 is not None and B1 is not None:
                self.BPlot[e] = (B0, B1)

            print(
                f"Even Mark {e}: "
                f"T={e in self.TPlot}, "
                f"N={e in self.NPlot}, "
                f"B={e in self.BPlot}"
            )

    # --------------------------------------------------------

    # --------------------------------------------------------
    # The result of the 4-plane method for the entire dv
    # is now displaied:

    def print_points(self):

        print("\n--- Coordinates ---")

        for m in sorted(self.points):

            p = self.points[m]
            e = self.errors[m]

            if p is None:
                print(f"{m}: FAILED")
            else:
                print(
                    f"{m}: "
                    f"("
                    f" {p[0]:.2f}, "
                    f" {p[1]:.2f}, "
                    f" {p[2]:.2f} "
                    f")  err={e:.3f}"
                )

    # END of the INTRINSIC RECEPTION of a coordinate frame
    # for the Marks in dv.
    # --------------------------------------------------------

    def set_axes_equal(self, ax, points=None, padding=0.01):

        if points is None or len(points) == 0:
            return

        points = np.asarray(points, dtype=float)

        xmin, ymin, zmin = np.min(points, axis=0)
        xmax, ymax, zmax = np.max(points, axis=0)

        x_mid = 0.5 * (xmin + xmax)
        y_mid = 0.5 * (ymin + ymax)
        z_mid = 0.5 * (zmin + zmax)

        x_range = xmax - xmin
        y_range = ymax - ymin
        z_range = zmax - zmin

        max_range = max(x_range, y_range, z_range)

        if max_range < 1e-9:
            max_range = 1.0

        radius = 0.5 * max_range * (1.0 + padding)

        ax.set_xlim3d(x_mid - radius, x_mid + radius)
        ax.set_ylim3d(y_mid - radius, y_mid + radius)
        ax.set_zlim3d(z_mid - radius, z_mid + radius)

        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    # --------------------------------------------------------

    def _selected_marks_and_points(
            self, mark_from=None, mark_to=None
        ):

        return [
            (m, p)
            for m, p in sorted(self.points.items())
            if p is not None
            and (mark_from is None or m >= mark_from)
            and (mark_to is None or m <= mark_to)
        ]

    # --------------------------------------------------------

    def _selected_triads(self, mark_from=None, mark_to=None):

        selected = []

        for i in sorted(self.triads):

            m0, m1, m2 = self.triads[i]

            if mark_from is not None and m0 < mark_from:
                continue
            if mark_to is not None and m2 > mark_to:
                continue

            selected.append(i)

        return selected

    # --------------------------------------------------------

    def _is_differential_interval(
            self, mark_from=None, mark_to=None
        ):

        # In this program, a Differential Interval is represented by
        # an explicitly selected part of the already analyzed Curve.
        # It must contain at least one complete Triad, hence at least
        # one complete Arc.

        if mark_from is None and mark_to is None:
            return False

        return len(self._selected_triads(mark_from, mark_to)) >= 1

    # --------------------------------------------------------

    def _curve_sample_points_for_avoidance(
            self, mark_from=None, mark_to=None
        ):
        """
        Collect points already lying on the plotted curve.
        These are used only for smart label placement, so that
        Mark numbers are preferably placed away from the curve
        path.
        """

        curve_pts = []

        for m, p in self._selected_marks_and_points(
            mark_from, mark_to
        ):
            curve_pts.append(np.asarray(p, dtype=float))

        for i in self._selected_triads(mark_from, mark_to):
            m0, m1, m2 = self.triads[i]

            curve_pts.append(np.asarray(
                self.points[m0], dtype=float)
            )

            for p in self.extra_points[i]:
                curve_pts.append(np.asarray(p, dtype=float))

            curve_pts.append(np.asarray(
                self.points[m2], dtype=float)
            )

        if not curve_pts:
            return np.empty((0, 3), dtype=float)

        return np.array(curve_pts, dtype=float)

    # --------------------------------------------------------

    def _draw_smart_mark_labels(
        self,
        ax,
        marks_and_pts,
        mark_from=None,
        mark_to=None,
        show_connectors=True,
        base_offset=12.0,
        fontsize=8,
    ):
        """
        Draw Mark numbers with automatic displacement.

        Aim:
        1) reduce mutual superposition of Mark numbers;
        2) reduce superposition of connector segments and
           labels with the green Curve;
        3) improve plots when two or more Marks are very
           close.

        This is a geometric heuristic, not a true 2D
        screen-space collision solver. It works in the 3D
        coordinates before Matplotlib projection.
        """

        if not marks_and_pts:
            return

        directions = [
            np.array([ 1,  1,  1], dtype=float),
            np.array([ 1, -1,  1], dtype=float),
            np.array([-1,  1,  1], dtype=float),
            np.array([-1, -1,  1], dtype=float),
            np.array([ 1,  1, -1], dtype=float),
            np.array([ 1, -1, -1], dtype=float),
            np.array([-1,  1, -1], dtype=float),
            np.array([-1, -1, -1], dtype=float),

            # Extra directions help when many Marks are
            # superposed or almost superposed.
            np.array([ 1,  0,  0], dtype=float),
            np.array([-1,  0,  0], dtype=float),
            np.array([ 0,  1,  0], dtype=float),
            np.array([ 0, -1,  0], dtype=float),
            np.array([ 0,  0,  1], dtype=float),
            np.array([ 0,  0, -1], dtype=float),
        ]

        curve_pts = self._curve_sample_points_for_avoidance(
            mark_from=mark_from,
            mark_to=mark_to
        )

        used_label_positions = []

        for idx, (m, p) in enumerate(marks_and_pts):

            p = np.asarray(p, dtype=float)

            best_q = None
            best_score = None

            # Increasing radii.  The first free-looking
            # place wins through the score minimization.
            for scale in [1.0, 1.4, 1.8, 2.3, 2.9, 3.6]:

                for d in directions:

                    d = d / np.linalg.norm(d)
                    q = p + base_offset * scale * d

                    score = 0.0

                    # 1) Avoid previously placed labels.
                    for old_q in used_label_positions:
                        dist = np.linalg.norm(q - old_q)
                        score += 1500.0 / max(dist, 1e-6)

                    # 2) Avoid placing labels too close to the
                    # curve path.
                    if len(curve_pts) > 0:
                        distances_to_curve = np.linalg.norm(
                            curve_pts - q,
                            axis=1
                        )
                        min_curve_dist = np.min(
                            distances_to_curve
                        )
                        score += 700.0 / max(
                            min_curve_dist, 1e-6
                        )

                    # 3) Prefer not-too-long connector
                    # segments.
                    score += 0.03 * np.linalg.norm(q - p)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_q = q

            q = best_q
            used_label_positions.append(q)

            if show_connectors:
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    [p[2], q[2]],
                    color="black",
                    linewidth=0.35,
                    alpha=0.45
                )

            ax.text(
                q[0], q[1], q[2],
                str(m),
                fontsize=fontsize,
                color="black"
            )

    # --------------------------------------------------------

    def _draw_green_curve_background(
        self,
        ax,
        mark_from=None,
        mark_to=None,
        show_marks=True,
        show_labels=True,
        show_connectors=True,
        auto_clean=True,
    ):

        marks_and_pts = self._selected_marks_and_points(
            mark_from, mark_to
        )

        if len(marks_and_pts) < 2:
            print("Not enough points in selected interval.")
            return

        plotted_points = []

        differential_interval = self._is_differential_interval(
            mark_from,
            mark_to
        )

        if auto_clean and differential_interval:
            show_marks = False
            show_labels = False
            show_connectors = False

        if show_marks:
            pts = np.array([p for m, p in marks_and_pts])

            plotted_points.extend(pts)

            ax.plot(
                pts[:,0],
                pts[:,1],
                pts[:,2],
                'go',
                markersize=1.5
            )
        else:
            pts = np.array([p for m, p in marks_and_pts])
            plotted_points.extend(pts)

        # True circular Arcs from complete selected Triads.
        for i in self._selected_triads(mark_from, mark_to):

            m0, m1, m2 = self.triads[i]

            arc_pts = [self.points[m0]]
            arc_pts.extend(self.extra_points[i])
            arc_pts.append(self.points[m2])

            arc_pts = np.array(arc_pts)

            plotted_points.extend(arc_pts)

            ax.plot(
                arc_pts[:,0],
                arc_pts[:,1],
                arc_pts[:,2],
                'g-',
                linewidth=0.75,
            )

        if show_labels:
            self._draw_smart_mark_labels(
                ax,
                marks_and_pts,
                show_connectors=show_connectors
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        self.set_axes_equal(
            ax,
            points=plotted_points,
            padding=0.01
        )

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='z', which='major', labelsize=8)

    # --------------------------------------------------------

    def plot_all(self, mark_from=None, mark_to=None):

        marks_and_pts = self._selected_marks_and_points(
            mark_from, mark_to
        )

        if len(marks_and_pts) < 2:
            print("Not enough points.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        self._draw_green_curve_background(
            ax,
            mark_from=mark_from,
            mark_to=mark_to,
            show_marks=True,
            show_labels=True,
            show_connectors=True,
            auto_clean=True,
        )

        if mark_from is None and mark_to is None:
            filename = "curve.png"
        else:
            filename = f"curve_{mark_from}_{mark_to}.png"

        if self.curve_name:
            ax.text2D(
                0.985, 0.995,
                self.curve_name,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="black"
            )
        #if self.curve_name:
        #    ax.text2D(
        #        1.12, 0.01,
        #        self.curve_name,
        #        transform=ax.transAxes,
        #        ha="right",
        #        va="bottom",
        #        fontsize=9,
        #        color="black"
        #    )
        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02
        )
        plt.close()

        print(f"✔ {filename} saved")

    # --------------------------------------------------------

    def _plot_frame_vectors(
        self,
        vectors,
        filename,
        mark_from=None,
        mark_to=None,
        frame_name="",
    ):

        marks_and_pts = self._selected_marks_and_points(
            mark_from, mark_to
        )

        if len(marks_and_pts) < 2:
            print("Not enough points.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Same background, the green Curve.
        # For TPlot/NPlot/BPlot we intentionally hide:
        # - Mark dots,
        # - Mark numbers,
        # - connector segments.
        # The origins of the incoming/outgoing vectors are
        # sufficient to locate the Marks and the plot is much
        # cleaner.
        self._draw_green_curve_background(
            ax,
            mark_from=mark_from,
            mark_to=mark_to,
            show_marks=False,
            show_labels=False,
            show_connectors=False,
            auto_clean=True,
        )

        selected_marks = {
            m
            for m, p in marks_and_pts
        }

        # Same length of represented unit vectors.
        # Incoming vector is orange; outgoing vector is purple.
        for e, (v0, v1) in sorted(vectors.items()):

            if e not in selected_marks:
                continue

            if e not in self.points:
                continue

            p = self.points[e]
            if p is None:
                continue

            for v, hue in [
                (v0, "darkorange"),   # incoming
                (v1, "purple")        # outgoing
            ]:
                
                vv = normalize(v)
                if vv is None:
                    continue

                vv = 2.00 * VEC_LEN * vv

                # ax.plot(
                #     [p[0], p[0] + vv[0]],
                #     [p[1], p[1] + vv[1]],
                #     [p[2], p[2] + vv[2]],
                #     color=hue,
                #     linewidth=1.2
                # )
                ax.quiver(
                    p[0],
                    p[1],
                    p[2],
                    vv[0],
                    vv[1],
                    vv[2],
                    color=hue,
                    linewidth=1.1,
                    arrow_length_ratio=0.3
                )

        self.set_axes_equal(ax)

        if mark_from is not None or mark_to is not None:
            stem = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1]
            filename = f"{stem}_{mark_from}_{mark_to}.{ext}"

        label = self.curve_name

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------
        label = f"{self.curve_name}   Vector {frame_name}"

        ax.text2D(
            1.0, 0.970,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="black"
        )

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        ax.text2D(
            1.0, 0.010,
            "At Even Marks",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=4,
            color="black"
        )

        ax.text2D(
            1.0, -0.015,
            "Incoming Unit Vector",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=4,
            color="darkorange"
        )

        ax.text2D(
            1.0, -0.037,
            "Outgoing Unit Vector",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=4,
            color="purple"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02
        )
        plt.close()
        
        print(f"✔ {filename} saved")

    # --------------------------------------------------------

    def plot_fs_comparisons(
            self, mark_from=None, mark_to=None
        ):

        # Generation of the three plots for the Curve:
        # one with T, one with N, one with B.
        # In these plots the Mark dots, Mark numbers and
        # connector segments are hidden, so the unit vectors
        # remain readable.

        self.plot_T(mark_from=mark_from, mark_to=mark_to)
        self.plot_N(mark_from=mark_from, mark_to=mark_to)
        self.plot_B(mark_from=mark_from, mark_to=mark_to)


    # --------------------------------------------------------

    def print_triads(self):

        print("\n--- Triads ---")

        if not self.triads:
            print("No complete triads available.")
            return

        for i in sorted(self.triads):
            m0, m1, m2 = self.triads[i]
            print(f"A_{i}: triad=({m0},{m1},{m2})")

    # --------------------------------------------------------

    def print_arcs(self):

        print("\n--- Arcs, Radii, Curvatures ---")

        if not self.arcs:
            print("No arcs available.")
            return

        for i in sorted(self.arcs):
            arc = self.arcs[i]
            center = arc["center"]
            radius = arc["radius"]
            curvature = self.curvatures.get(i, None)

            if center is None or not np.isfinite(radius):
                print(f"A_{i}: collinear or undefined arc")
                continue

            print(
                f"A_{i}: marks={arc['marks']}  "
                f"center=("
                f"{center[0]:.2f}, "
                f"{center[1]:.2f}, "
                f"{center[2]:.2f}"
                f")  "
                f"radius={radius:.3f}  k={curvature:.6f}"
            )

    # --------------------------------------------------------

    def check_distances(self):

        print("\n--- Distance Check ---")
        print(
            f"Mark   q0_err    q1_err    q2_err    q3_err    "
            f"mean_abs_err"
        )

        for m in sorted(self.points):

            p = self.points[m]
            if p is None:
                print(f"{m:4d}: FAILED")
                continue

            measured = self.dv[m]
            diffs = []

            for i in range(4):
                reconstructed = np.linalg.norm(p - self.qs[i])
                diffs.append(reconstructed - measured[i])

            mean_abs = np.mean(np.abs(diffs))

            print(
                f"{m:4d} "
                f"{diffs[0]:8.3f} "
                f"{diffs[1]:8.3f} "
                f"{diffs[2]:8.3f} "
                f"{diffs[3]:8.3f} "
                f"{mean_abs:12.3f}"
            )

    # --------------------------------------------------------

    def exact_frenet_serret_frames(self):

        print("\n--- Exact Frenet-Serret Frames ---")

        if not self.arc_frames:
            print("No Frenet-Serret frames available.")
            return

        for i in sorted(self.arc_frames):

            frames = self.arc_frames[i]
            if frames is None:
                print(f"A_{i}: undefined frame")
                continue

            print(f"\nA_{i}: triad={self.triads[i]}")

            for side in ("left", "middle", "right"):
                if side not in frames:
                    continue

                fs = frames[side]
                print(f"  {side} endpoint:")

                for name in ("T", "N", "B"):
                    v = fs[name]
                    if v is None:
                        print(f"    {name}: undefined")
                    else:
                        print(
                            f"    {name}: "
                            f"("
                            f"{v[0]: .6f}, "
                            f"{v[1]: .6f}, "
                            f"{v[2]: .6f}"
                            f")"
                        )

    # --------------------------------------------------------

    def plot_frenet_serret_comparisons(self, mark_from=None, mark_to=None):

        self.plot_fs_comparisons(mark_from=mark_from, mark_to=mark_to)

    # --------------------------------------------------------

    def plot_T(self, mark_from=None, mark_to=None):

        self._plot_frame_vectors(
            self.TPlot,
            "Plot_T.png",
            mark_from=mark_from,
            mark_to=mark_to,
            frame_name="T"
        )

    # --------------------------------------------------------

    def plot_N(self, mark_from=None, mark_to=None):

        self._plot_frame_vectors(
            self.NPlot,
            "Plot_N.png",
            mark_from=mark_from,
            mark_to=mark_to,
            frame_name="N"
        )

    # --------------------------------------------------------

    def plot_B(self, mark_from=None, mark_to=None):

        self._plot_frame_vectors(
            self.BPlot,
            "Plot_B.png",
            mark_from=mark_from,
            mark_to=mark_to,
            frame_name="B"
        )

    # --------------------------------------------------------

    def make_movie(
            self, filename="curve.mp4",
            mark_from=None, mark_to=None
        ):

        try:
            import imageio
        except:
            print("Install imageio")
            return

        marks_and_pts = self._selected_marks_and_points(
            mark_from, mark_to
        )

        if len(marks_and_pts) < 2:
            print("Not enough points.")
            return

        if mark_from is not None or mark_to is not None:
            stem = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1]
            filename = f"{stem}_{mark_from}_{mark_to}.{ext}"

        frames = []

        for a in np.linspace(0, 360, 180):

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            self._draw_green_curve_background(
                ax,
                mark_from=mark_from,
                mark_to=mark_to,
                show_marks=False,
                show_labels=False,
                show_connectors=False,
                auto_clean=True,
            )

            ax.view_init(30, a)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[:,:,:3])

            plt.close(fig)

        imageio.mimsave(filename, frames, fps=24)

        print(f"✔ {filename} saved")

# ============================================================
# Data loading
# ============================================================

def load_data(data_name):

    path = Path("data") / f"{data_name}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: "
            f"{data_name}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

        curve_name = data.get("curve_name", data_name)
    ref_distances = tuple(data["ref_distances"])
    handedness = data.get("handedness", "right")
    tolerance = data.get("tolerance", 1.0)

    dv = {
        int(k): v
        for k, v in data["dv"].items()
    }

    return ref_distances, dv, handedness, tolerance, curve_name


# ============================================================
# Action dispatcher
# ============================================================

def run_action(cp, action, mark_from=None, mark_to=None):

    if action == "coordinates":
        cp.print_points()

    elif action == "triads":
        cp.print_triads()

    elif action == "arcs":
        cp.print_arcs()
        cp.plot_all(mark_from=mark_from, mark_to=mark_to)

    elif action == "check":
        cp.check_distances()

    elif action == "plot":
        cp.plot_all(mark_from=mark_from, mark_to=mark_to)

    elif action == "movie":
        cp.make_movie(
            "curve.mp4",
            mark_from=mark_from,
            mark_to=mark_to
        )

    elif action == "fs":
        cp.exact_frenet_serret_frames()

    elif action == "fs_plots":
        cp.plot_frenet_serret_comparisons(
            mark_from=mark_from,
            mark_to=mark_to
        )

    elif action == "plot_t":
        cp.plot_T(mark_from=mark_from, mark_to=mark_to)

    elif action == "plot_n":
        cp.plot_N(mark_from=mark_from, mark_to=mark_to)

    elif action == "plot_b":
        cp.plot_B(mark_from=mark_from, mark_to=mark_to)

    elif action == "all":
        cp.print_points()
        cp.print_triads()
        cp.print_arcs()
        cp.check_distances()
        cp.exact_frenet_serret_frames()
        cp.plot_all(mark_from=mark_from, mark_to=mark_to)
        cp.plot_frenet_serret_comparisons(
            mark_from=mark_from,
            mark_to=mark_to
        )
        cp.make_movie(
            "curve.mp4",
            mark_from=mark_from,
            mark_to=mark_to
        )

    else:
        raise ValueError(f"Unknown action: {action}")

    def _stamp_curve_name(self, fig, curve_name):

        if not curve_name:
            return

        fig.text(
            0.985, 0.015,
            curve_name,
            ha="right",
            va="bottom",
            fontsize=9,
            color="black"
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run CPlanes from an external data file."
    )

    parser.add_argument(
        "data",
        help=
            f"Name of the data file without extension, "
            f"e.g. curve7"
    )

    parser.add_argument(
        "actions",
        nargs="+",
        choices=[
            "coordinates",
            "triads",
            "arcs",
            "check",
            "plot",
            "movie",
            "fs",
            "fs_plots",
            "plot_t",
            "plot_n",
            "plot_b",
            "all"
        ],
        help="One or more outputs/actions to produce."
    )

    parser.add_argument(
        "--from-mark",
        type=int,
        default=None,
        help="First Mark to plot, inclusive."
    )

    parser.add_argument(
        "--to-mark",
        type=int,
        default=None,
        help="Last Mark to plot, inclusive."
    )

    args = parser.parse_args()

    (
        ref_distances,
        dv,
        handedness,
        tolerance,
        curve_name,
    ) = load_data(args.data)

    cp = CPlanes(
        ref_distances,
        dv,
        tol=tolerance,
        handedness=handedness,
        curve_name=curve_name
    )

    cp.reconstruct()

    for action in args.actions:
        run_action(
            cp,
            action,
            mark_from=args.from_mark,
            mark_to=args.to_mark
        )
    