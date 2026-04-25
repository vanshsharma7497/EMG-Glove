"""
core/letter_templates.py
-------------------------
Defines handwriting stroke templates for all 26 letters, digits 0–9,
and 13 common symbols (48 templates total).
Each template is a normalised (x, y) path in a 0..1 bounding box.

Multi-stroke characters (f, k, t, x) are approximated as their most
distinctive single stroke. The context agent layer compensates for the
resulting ambiguity by providing a language-model prior on likely characters.

Also provides the recogniser: given an incoming stroke from the
StrokeProcessor, find the closest matching template.

This is the direct equivalent of GESTURE_PROFILES in emg_simulator.py —
a dictionary of named shapes that the classifier matches against.

When real hardware data is available, templates can be replaced with
recorded real strokes (averaged across multiple repetitions) for better
accuracy. The recogniser code stays identical.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Template builder helpers
# ---------------------------------------------------------------------------

def _interpolate(waypoints, n_points=64):
    """
    Takes a list of (x, y) waypoints and returns a smooth path of n_points
    by linearly interpolating between each consecutive pair of waypoints.

    This lets us define a letter with just a handful of turning points
    rather than specifying every single coordinate by hand.

    waypoints : list of (x, y) tuples — key turning points of the stroke
    n_points  : total number of points in the output path
    """
    waypoints = np.array(waypoints, dtype=float)
    n_segments = len(waypoints) - 1

    # Distribute points evenly across segments
    pts_per_segment = max(2, n_points // n_segments)

    path = []
    for i in range(n_segments):
        start = waypoints[i]
        end   = waypoints[i + 1]
        # linspace from start to end — exclude last point to avoid duplicates
        # except on the final segment
        include_end = (i == n_segments - 1)
        t = np.linspace(0, 1, pts_per_segment, endpoint=include_end)
        for ti in t:
            path.append(start + ti * (end - start))

    path = np.array(path)

    # Normalise to 0..1 bounding box (same as StrokeProcessor._end_stroke)
    x_min, y_min = path[:, 0].min(), path[:, 1].min()
    x_max, y_max = path[:, 0].max(), path[:, 1].max()
    x_range = x_max - x_min if x_max - x_min > 1e-6 else 1.0
    y_range = y_max - y_min if y_max - y_min > 1e-6 else 1.0

    path[:, 0] = (path[:, 0] - x_min) / x_range
    path[:, 1] = (path[:, 1] - y_min) / y_range

    return path   # shape: (n_points_approx, 2)


def _arc(cx, cy, radius, start_angle_deg, end_angle_deg, n=20):
    """
    Generates points along a circular arc — used for curved letters like c, o, u.

    cx, cy           : centre of the arc
    radius           : radius of the arc
    start_angle_deg  : starting angle in degrees (0 = right, 90 = down)
    end_angle_deg    : ending angle in degrees
    n                : number of points along the arc
    """
    angles = np.linspace(np.radians(start_angle_deg),
                         np.radians(end_angle_deg), n)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return list(zip(xs, ys))


# ---------------------------------------------------------------------------
# Letter template definitions
# ---------------------------------------------------------------------------
# Convention:
#   (0, 0) = top-left of bounding box
#   (1, 0) = top-right
#   (0, 1) = bottom-left
#   (1, 1) = bottom-right
#   Y increases downward (screen coordinates — same as cursor y-axis)
#
# Each entry in LETTER_TEMPLATES is a list of (x, y) waypoints.
# _interpolate() converts these into a smooth dense path.
# ---------------------------------------------------------------------------

def _build_templates():
    templates = {}

    # ── l — downstroke with a serif foot at the bottom ────────────────────
    # The horizontal foot makes it distinct from c (a curve) and i (shorter
    # foot that goes both ways). Under noise, the foot's horizontal segment
    # is enough to prevent confusion with any arc-shaped letter.
    templates["l"] = _interpolate([
        (0.5, 0.0),   # top centre
        (0.5, 1.0),   # bottom centre
        (0.7, 1.0),   # foot — short rightward serif only (distinguishes from i)
    ])

    # ── i — downstroke with a slight leftward serif at the base ───────────
    # Needs to be visually distinct from l. We add a small bottom foot.
    templates["i"] = _interpolate([
        (0.5, 0.0),    # top
        (0.5, 1.0),    # bottom
        (0.25, 1.0),   # short foot left
        (0.75, 1.0),   # short foot right
    ])

    # ── v — down-right diagonal then up-right diagonal ────────────────────
    templates["v"] = _interpolate([
        (0.0, 0.0),   # top left
        (0.5, 1.0),   # bottom centre (the point of the V)
        (1.0, 0.0),   # top right
    ])

    # ── u — down, curve at bottom, back up ────────────────────────────────
    bottom_arc = _arc(0.5, 0.75, 0.25, 180, 0, n=16)   # bottom curve
    templates["u"] = _interpolate(
        [(0.25, 0.0), (0.25, 0.75)] +   # left downstroke
        bottom_arc +                      # bottom curve
        [(0.75, 0.75), (0.75, 0.0)]      # right upstroke
    )

    # ── n — two humps (mirrored u shape — arcs going upward) ──────────────
    hump1 = _arc(0.25, 0.35, 0.2, 180, 0, n=12)   # first arch (top arc, opens downward)
    hump2 = _arc(0.75, 0.35, 0.2, 180, 0, n=12)
    templates["n"] = _interpolate(
        [(0.05, 1.0), (0.05, 0.35)] +   # first upstroke
        hump1 +                           # first arch
        [(0.45, 0.35), (0.45, 1.0),      # down + second upstroke
         (0.45, 0.35)] +
        hump2 +                           # second arch
        [(0.95, 0.35), (0.95, 1.0)]      # final downstroke
    )

    # ── c — open arc, starts mid-right, sweeps left and down ──────────────
    arc_pts = _arc(0.5, 0.5, 0.45, 330, 30, n=30)
    # 330° to 30° going anti-clockwise = open on the right side
    templates["c"] = _interpolate(arc_pts)

    # ── o — closed oval ────────────────────────────────────────────────────
    arc_pts = _arc(0.5, 0.5, 0.45, 270, 270 + 360, n=40)
    templates["o"] = _interpolate(arc_pts)

    # ── e — like c but with a horizontal bar through the middle ───────────
    # Approximated as: horizontal bar left-to-right at midpoint, then arc
    bar   = [(0.15, 0.5), (0.85, 0.5)]
    curve = _arc(0.5, 0.55, 0.4, 0, 300, n=28)
    templates["e"] = _interpolate(bar + curve)

    # ── a — small oval on right + downstroke on right ─────────────────────
    oval  = _arc(0.38, 0.5, 0.32, 270, 270 + 330, n=28)
    templates["a"] = _interpolate(
        oval + [(0.7, 0.18), (0.7, 1.0)]   # close oval then downstroke
    )

    # ── r — short downstroke, then a single small hump ────────────────────
    hump = _arc(0.45, 0.45, 0.3, 180, 0, n=14)
    templates["r"] = _interpolate(
        [(0.15, 1.0), (0.15, 0.45)] +   # upstroke
        hump +                            # small arch
        [(0.75, 0.45)]                   # end of hump, pen lifts here
    )

    # ── s — reverse-c on top, c on bottom ─────────────────────────────────
    top_arc    = _arc(0.5, 0.3, 0.28, 30,  210, n=16)   # upper reverse-arc
    bottom_arc_s = _arc(0.5, 0.7, 0.28, 210, 30,  n=16)   # lower forward-arc
    templates["s"] = _interpolate(top_arc + bottom_arc_s)

    # =========================================================================
    # NEW LETTERS (b – z, excluding those already defined above)
    # =========================================================================

    # ── b — downstroke from top, clockwise loop on right at bottom ───────────
    # Long downstroke distinguishes it from p (which starts at midpoint).
    # The loop sits in the lower half — distinguishes from d (loop in upper half).
    loop_b = _arc(0.55, 0.75, 0.28, 270, 270 + 330, n=24)
    templates["b"] = _interpolate(
        [(0.27, 0.0), (0.27, 1.0),          # full downstroke
         (0.27, 0.47)] +                      # back up to loop start
        loop_b
    )

    # ── d — oval on left, tall upstroke on right ──────────────────────────────
    # Mirror of b's logic but oval sits in the upper-mid area and the
    # upstroke goes to the very top. Distinguishes from a (shorter upstroke).
    oval_d = _arc(0.38, 0.5, 0.30, 270, 270 + 330, n=24)
    templates["d"] = _interpolate(
        oval_d +
        [(0.68, 0.20), (0.68, 0.0),          # upstroke to top
         (0.68, 1.0)]                          # back down to baseline
    )

    # ── f — curved hook at top-left, long downstroke ──────────────────────────
    # Single-stroke approximation: start at hook, curve right, then down.
    # Crossbar is omitted (second stroke) — context agent handles ambiguity.
    hook_f = _arc(0.5, 0.22, 0.22, 180, 300, n=16)   # hook curves right at top
    templates["f"] = _interpolate(
        hook_f +
        [(0.5, 0.22), (0.5, 1.0)]            # downstroke to baseline
    )

    # ── g — full oval, then descending tail curving left ──────────────────────
    # Tail drops below the baseline (y > 1.0 before normalisation).
    # Distinguishes from q whose tail curves right.
    oval_g = _arc(0.38, 0.45, 0.30, 270, 270 + 360, n=28)
    templates["g"] = _interpolate(
        oval_g +
        [(0.68, 0.15), (0.68, 1.2),          # right side of oval, tail down
         (0.45, 1.35), (0.2, 1.2)]            # tail curves left below baseline
    )

    # ── h — long downstroke, hump branching right from midpoint ───────────────
    # Start from top, go down, come back up to midpoint, arch right, down again.
    hump_h = _arc(0.6, 0.5, 0.22, 180, 0, n=14)
    templates["h"] = _interpolate(
        [(0.18, 0.0), (0.18, 1.0),           # full downstroke
         (0.18, 0.5)] +                        # back up to arch start
        hump_h +
        [(0.82, 0.5), (0.82, 1.0)]            # right downstroke
    )

    # ── j — straight downstroke curving left below baseline ───────────────────
    # Similar to i but no foot at top and tail curves below the baseline.
    templates["j"] = _interpolate([
        (0.5, 0.0),    # top
        (0.5, 1.0),    # baseline
        (0.5, 1.15),   # below baseline
        (0.3, 1.3),    # curve left
        (0.15, 1.2),   # hook end
    ])

    # ── k — downstroke, diagonal in to midpoint, diagonal out ─────────────────
    # Single-stroke zigzag approximation. The sharp angle at the meeting
    # point is what distinguishes k from h (which has a smooth arch).
    templates["k"] = _interpolate([
        (0.2, 0.0),    # top of downstroke
        (0.2, 1.0),    # bottom of downstroke
        (0.2, 0.5),    # back up to midpoint
        (0.8, 0.0),    # diagonal up-right
        (0.2, 0.5),    # back to meeting point
        (0.8, 1.0),    # diagonal down-right
    ])

    # ── m — three humps (like n but one extra arch) ────────────────────────────
    hump_m1 = _arc(0.22, 0.38, 0.17, 180, 0, n=10)
    hump_m2 = _arc(0.55, 0.38, 0.17, 180, 0, n=10)
    hump_m3 = _arc(0.82, 0.38, 0.13, 180, 0, n=10)
    templates["m"] = _interpolate(
        [(0.05, 1.0), (0.05, 0.38)] +        # first upstroke
        hump_m1 +
        [(0.38, 0.38), (0.38, 1.0),
         (0.38, 0.38)] +                       # down and up
        hump_m2 +
        [(0.72, 0.38), (0.72, 1.0),
         (0.72, 0.38)] +                       # down and up
        hump_m3 +
        [(0.95, 0.38), (0.95, 1.0)]           # final downstroke
    )

    # ── p — downstroke below baseline, loop on upper-right ────────────────────
    # Starts at midpoint (not top like b) and tail drops below baseline.
    # Loop sits in the upper half — distinguishes from b (lower half loop).
    loop_p = _arc(0.55, 0.35, 0.27, 270, 270 + 330, n=22)
    templates["p"] = _interpolate(
        [(0.28, 0.08), (0.28, 1.3),          # downstroke through and below baseline
         (0.28, 0.08)] +                       # back up to loop start
        loop_p
    )

    # ── q — full oval, then tail dropping right below baseline ────────────────
    # Mirror of g. Tail goes down-right instead of down-left.
    oval_q = _arc(0.38, 0.45, 0.30, 270, 270 + 360, n=28)
    templates["q"] = _interpolate(
        oval_q +
        [(0.68, 0.15), (0.68, 1.2),          # right side, tail down
         (0.85, 1.35), (0.95, 1.2)]           # tail curves right below baseline
    )

    # ── t — straight downstroke (crossbar is second stroke, omitted) ──────────
    # Taller than i and l because t ascends above the midline.
    # Foot distinguishes from a bare downstroke — small rightward serif.
    templates["t"] = _interpolate([
        (0.5, 0.0),    # top
        (0.5, 1.0),    # baseline
        (0.65, 1.0),   # small foot right (distinguishes from l's longer foot)
    ])

    # ── w — four-point zigzag (two valleys, like double-v) ────────────────────
    templates["w"] = _interpolate([
        (0.0,  0.0),   # top left
        (0.25, 1.0),   # first valley
        (0.5,  0.4),   # first rise (not full height — w is squat)
        (0.75, 1.0),   # second valley
        (1.0,  0.0),   # top right
    ])

    # ── x — one diagonal only (second diagonal is second stroke, omitted) ─────
    # Top-left to bottom-right. Context agent resolves the stroke ambiguity.
    templates["x"] = _interpolate([
        (0.1, 0.0),    # top left
        (0.9, 1.0),    # bottom right
    ])

    # ── y — two legs meeting at centre, right leg continues below baseline ─────
    # Left leg comes in from top-left, right leg from top-right, they meet at
    # centre and the stroke continues down and below baseline.
    templates["y"] = _interpolate([
        (0.1,  0.0),   # top left
        (0.5,  0.6),   # centre meeting point
        (0.9,  0.0),   # top right
        (0.5,  0.6),   # back to centre
        (0.5,  1.0),   # baseline
        (0.35, 1.3),   # tail curves left below baseline
    ])

    # ── z — horizontal top, diagonal down-left, horizontal base ───────────────
    templates["z"] = _interpolate([
        (0.1,  0.0),   # top left
        (0.9,  0.0),   # top right
        (0.1,  1.0),   # diagonal to bottom left
        (0.9,  1.0),   # bottom right
    ])

    # =========================================================================
    # DIGITS 0 – 9
    # =========================================================================

    # ── 0 — closed oval, slightly wider than o ────────────────────────────────
    arc_0 = _arc(0.5, 0.5, 0.48, 270, 270 + 360, n=40)
    templates["0"] = _interpolate(arc_0)

    # ── 1 — short diagonal lead-in then straight downstroke ───────────────────
    templates["1"] = _interpolate([
        (0.3, 0.15),   # lead-in start (top-left)
        (0.5, 0.0),    # top of downstroke
        (0.5, 1.0),    # bottom
    ])

    # ── 2 — arc from top-right curving left-down, then horizontal base ─────────
    arc_2 = _arc(0.5, 0.28, 0.28, 300, 180, n=20)   # top arc, opens leftward
    templates["2"] = _interpolate(
        arc_2 +
        [(0.15, 0.55), (0.1, 1.0),           # diagonal sweep down-left
         (0.9, 1.0)]                           # horizontal base
    )

    # ── 3 — upper right arc then lower right arc ──────────────────────────────
    arc_3_top = _arc(0.5, 0.28, 0.30, 300, 90, n=16)   # upper bump, opens left
    arc_3_bot = _arc(0.5, 0.72, 0.30, 270, 60, n=16)   # lower bump, opens left
    templates["3"] = _interpolate(arc_3_top + arc_3_bot)

    # ── 4 — diagonal down-left, horizontal right, vertical down from top-right ─
    # Single continuous stroke: left leg → crossbar → right leg down.
    templates["4"] = _interpolate([
        (0.65, 0.0),   # top right
        (0.1,  0.65),  # diagonal down-left
        (0.9,  0.65),  # horizontal right (the crossbar)
        (0.65, 0.65),  # back to junction
        (0.65, 1.0),   # vertical downstroke
    ])

    # ── 5 — horizontal top-left, downstroke, belly curve right ────────────────
    belly_5 = _arc(0.5, 0.72, 0.30, 180, 60, n=20)   # lower belly, opens right
    templates["5"] = _interpolate(
        [(0.8, 0.0), (0.15, 0.0),            # horizontal top (right to left)
         (0.15, 0.42)] +                       # downstroke
        belly_5
    )

    # ── 6 — curve from top sweeping left-down into closed loop at bottom ───────
    loop_6 = _arc(0.5, 0.72, 0.28, 270, 270 + 360, n=28)   # closed bottom loop
    templates["6"] = _interpolate(
        [(0.75, 0.0), (0.4, 0.05),           # start top-right, sweep left
         (0.1,  0.35), (0.22, 0.44)] +        # curve down into loop entry
        loop_6
    )

    # ── 7 — horizontal top then diagonal down-left ─────────────────────────────
    templates["7"] = _interpolate([
        (0.1,  0.0),   # top left
        (0.9,  0.0),   # top right
        (0.25, 1.0),   # diagonal down to bottom-left
    ])

    # ── 8 — figure-eight: upper loop then lower loop ──────────────────────────
    # Upper loop is smaller, lower loop larger — matches how 8 is written.
    loop_8_top = _arc(0.5, 0.32, 0.25, 270, 270 + 360, n=22)
    loop_8_bot = _arc(0.5, 0.70, 0.30, 90,  90  + 360, n=26)
    templates["8"] = _interpolate(loop_8_top + loop_8_bot)

    # ── 9 — closed loop at top, tail dropping below baseline ──────────────────
    loop_9 = _arc(0.45, 0.35, 0.30, 270, 270 + 360, n=28)
    templates["9"] = _interpolate(
        loop_9 +
        [(0.75, 0.35), (0.75, 1.0),          # right side of loop, tail down
         (0.6,  1.25), (0.4, 1.2)]            # slight curve at end of tail
    )

    # =========================================================================
    # SYMBOLS  (# and * excluded — too many strokes for prototype)
    # =========================================================================

    # ── @ — oval with tail wrapping clockwise around outside ──────────────────
    inner = _arc(0.45, 0.5, 0.25, 270, 270 + 330, n=22)   # inner a-style oval
    wrap  = _arc(0.5,  0.5, 0.45, 30,  30  + 330, n=28)   # outer wrap arc
    templates["@"] = _interpolate(inner + wrap)

    # ── & — loop from top, curves left-down, crosses itself, tail right ────────
    loop_amp = _arc(0.42, 0.32, 0.28, 270, 270 + 340, n=24)
    templates["&"] = _interpolate(
        loop_amp +
        [(0.14, 0.64), (0.15, 0.88),         # curve down-left
         (0.38, 1.0),  (0.65, 0.82),          # bottom curve right
         (0.85, 0.55), (0.5,  0.5),           # diagonal crossing up-left
         (0.85, 1.0)]                          # tail out to bottom-right
    )

    # ── + — horizontal then vertical, approximated as one L-turn stroke ────────
    # Written as: left → right along horizontal, then pivot and go down.
    templates["+"] = _interpolate([
        (0.1,  0.5),   # left of horizontal
        (0.9,  0.5),   # right of horizontal
        (0.5,  0.5),   # back to centre
        (0.5,  0.1),   # top of vertical
        (0.5,  0.9),   # bottom of vertical
    ])

    # ── - — short horizontal stroke ────────────────────────────────────────────
    templates["-"] = _interpolate([
        (0.1, 0.5),    # left
        (0.9, 0.5),    # right
    ])

    # ── = — two horizontals connected — approximated as Z-path ─────────────────
    # Top bar left-to-right, diagonal connector, bottom bar left-to-right.
    templates["="] = _interpolate([
        (0.1, 0.35),   # top bar left
        (0.9, 0.35),   # top bar right
        (0.1, 0.65),   # jump across to bottom bar left
        (0.9, 0.65),   # bottom bar right
    ])

    # ── / — diagonal bottom-left to top-right ──────────────────────────────────
    templates["/"] = _interpolate([
        (0.2, 1.0),    # bottom left
        (0.8, 0.0),    # top right
    ])

    # ── \ — diagonal top-left to bottom-right ──────────────────────────────────
    templates["\\"] = _interpolate([
        (0.2, 0.0),    # top left
        (0.8, 1.0),    # bottom right
    ])

    # ── ( — arc curving left (concave on right side) ───────────────────────────
    arc_open = _arc(0.6, 0.5, 0.4, 130, 230, n=20)
    templates["("] = _interpolate(arc_open)

    # ── ) — arc curving right (concave on left side, mirror of open paren) ─────
    arc_close = _arc(0.4, 0.5, 0.4, 310, 50, n=20)
    templates[")"] = _interpolate(arc_close)

    # ── [ — top horizontal, downstroke, bottom horizontal ─────────────────────
    templates["["] = _interpolate([
        (0.7, 0.0),    # top right
        (0.3, 0.0),    # top left
        (0.3, 1.0),    # downstroke
        (0.7, 1.0),    # bottom right
    ])

    # ── ] — mirror of [ ────────────────────────────────────────────────────────
    templates["]"] = _interpolate([
        (0.3, 0.0),    # top left
        (0.7, 0.0),    # top right
        (0.7, 1.0),    # downstroke
        (0.3, 1.0),    # bottom left
    ])

    # ── < — diagonal down-left then diagonal up-right (v rotated 90°) ─────────
    templates["<"] = _interpolate([
        (0.9, 0.1),    # top right
        (0.1, 0.5),    # left point
        (0.9, 0.9),    # bottom right
    ])

    # ── > — mirror of < ────────────────────────────────────────────────────────
    templates[">"] = _interpolate([
        (0.1, 0.1),    # top left
        (0.9, 0.5),    # right point
        (0.1, 0.9),    # bottom left
    ])

    return templates


# Build once at import time
LETTER_TEMPLATES = _build_templates()


# ---------------------------------------------------------------------------
# Recogniser
# ---------------------------------------------------------------------------

def _resample(path, n=64):
    """
    Resamples a path (list of (x,y) tuples OR numpy array) to exactly n points
    by interpolating along the cumulative arc length.

    This is important because two strokes of the same letter might have
    different numbers of recorded points (one was drawn faster, one slower).
    Resampling to a fixed length makes them directly comparable.
    """
    path = np.array(path)

    # Compute cumulative distance along the path
    diffs = np.diff(path, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-6:
        # Degenerate path — return repeated first point
        return np.tile(path[0], (n, 1))

    # Target distances: evenly spaced from 0 to total_length
    target = np.linspace(0, total_length, n)

    # Interpolate x and y separately
    resampled_x = np.interp(target, cum_lengths, path[:, 0])
    resampled_y = np.interp(target, cum_lengths, path[:, 1])

    return np.column_stack([resampled_x, resampled_y])


def recognise_letter(stroke, n_resample=64):
    """
    Given a completed stroke dict (from StrokeProcessor), find the closest
    matching letter template and return the recognised letter + confidence.

    stroke     : stroke dict from StrokeProcessor.update()
    n_resample : number of points to resample both stroke and templates to
                 before comparison — must be consistent

    Returns:
        letter     : str — the best matching letter
        confidence : float — 0.0 (no match) to 1.0 (perfect match)
        scores     : dict — distance score for every template (for debugging)
    """

    # Get the normalised stroke path from the processor output
    stroke_path = np.array(stroke["normalised"])

    # Resample to fixed length for fair comparison
    stroke_resampled = _resample(stroke_path, n=n_resample)

    best_letter   = None
    best_distance = float("inf")
    scores        = {}

    for letter, template in LETTER_TEMPLATES.items():
        # Resample template to same length
        template_resampled = _resample(template, n=n_resample)

        # Mean Euclidean distance between corresponding points
        # This is the simplest and most interpretable distance metric:
        # "on average, how far apart are the two paths at each step?"
        dist = np.mean(np.sqrt(
            (stroke_resampled[:, 0] - template_resampled[:, 0]) ** 2 +
            (stroke_resampled[:, 1] - template_resampled[:, 1]) ** 2
        ))

        scores[letter] = dist

        if dist < best_distance:
            best_distance = dist
            best_letter   = letter

    # Convert distance to a confidence score between 0 and 1.
    # A perfect match = distance 0 → confidence 1.0.
    # We use an exponential decay: confidence drops as distance grows.
    # The scale factor 3.0 was chosen so a "pretty good" match (~0.3 distance)
    # gives roughly 0.4 confidence, and a bad match gives near 0.
    confidence = float(np.exp(-best_distance * 3.0))

    return best_letter, confidence, scores


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Letter template self-test\n")
    print(f"  Templates defined: {sorted(LETTER_TEMPLATES.keys())}")
    print(f"  Points per template (before resample): varies\n")

    # ── Plot all templates ─────────────────────────────────────────────────
    letters = sorted(LETTER_TEMPLATES.keys())
    n_cols  = 5
    n_rows  = (len(letters) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5))
    fig.suptitle("Letter Templates — Normalised Stroke Paths", fontsize=13,
                 fontweight="bold")
    axes = axes.flatten()

    for i, letter in enumerate(letters):
        path = LETTER_TEMPLATES[letter]
        ax = axes[i]
        ax.plot(path[:, 0], path[:, 1], color="#4fc3f7", linewidth=2.0)
        ax.scatter(path[0, 0],  path[0, 1],  color="lime",   s=50, zorder=5,
                   label="start")
        ax.scatter(path[-1, 0], path[-1, 1], color="yellow", s=50, zorder=5,
                   label="end")
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.invert_yaxis()
        ax.set_title(f"  '{letter}'", fontsize=14, loc="left", pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")

    # Hide unused subplots
    for j in range(len(letters), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ── Test recogniser: match each template against itself ────────────────
    print("Recogniser self-match test (each template matched against itself):\n")
    print(f"  {'Input':<8} {'Best match':<12} {'Confidence':<12} {'Result'}")
    print("  " + "─" * 44)

    all_correct = True
    for letter, template in LETTER_TEMPLATES.items():
        # Wrap template as a fake stroke dict
        path = template
        fake_stroke = {
            "normalised": [(float(p[0]), float(p[1])) for p in path]
        }
        recognised, confidence, _ = recognise_letter(fake_stroke)
        correct = recognised == letter
        if not correct:
            all_correct = False
        mark = "✓" if correct else "✗"
        print(f"  {mark}  '{letter}'    →  '{recognised}'        "
              f"conf={confidence:.3f}")

    print()
    if all_correct:
        print("  All templates match themselves correctly.")
    else:
        print("  Some templates did not self-match — review waypoints.")

    # ── Test recogniser: add noise to templates and re-match ───────────────
    print("\nNoisy recognition test (templates + gaussian noise, σ=0.05):\n")
    print(f"  {'Input':<8} {'Best match':<12} {'Confidence':<12} {'Result'}")
    print("  " + "─" * 44)

    np.random.seed(42)
    correct_count = 0
    for letter, template in LETTER_TEMPLATES.items():
        noisy = template + np.random.normal(0, 0.05, template.shape)
        # Re-normalise after adding noise
        noisy[:, 0] = (noisy[:, 0] - noisy[:, 0].min()) / \
                      max(noisy[:, 0].max() - noisy[:, 0].min(), 1e-6)
        noisy[:, 1] = (noisy[:, 1] - noisy[:, 1].min()) / \
                      max(noisy[:, 1].max() - noisy[:, 1].min(), 1e-6)
        fake_stroke = {
            "normalised": [(float(p[0]), float(p[1])) for p in noisy]
        }
        recognised, confidence, _ = recognise_letter(fake_stroke)
        correct = recognised == letter
        if correct:
            correct_count += 1
        mark = "✓" if correct else "✗"
        print(f"  {mark}  '{letter}'    →  '{recognised}'        "
              f"conf={confidence:.3f}")

    print(f"\n  Noisy accuracy: {correct_count}/{len(LETTER_TEMPLATES)} "
          f"({100*correct_count//len(LETTER_TEMPLATES)}%)")