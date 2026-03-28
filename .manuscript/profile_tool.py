import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tap import Tap

from libs.checkpoints import Checkpoint

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ── CLI ──────────────────────────────────────────────────────────────────

class CLI(Tap):
    input: Path  # Path to a .npz checkpoint file

    def configure(self):
        self.add_argument(
            "input",
            type=Path,
            help="Path to a .npz checkpoint file.",
        )


# ── helpers ──────────────────────────────────────────────────────────────

def _intersections_along_line(x, U, p1, p2):
    """
    Given the spatial grid *x* (shape M) and the solution array *U*
    (shape K×M), compute where the straight line from *p1* to *p2*
    (in (x, y) data coordinates) crosses each curve u_n(x).

    Returns
    -------
    crossings : list[list[tuple[float, float, float]]]
        crossings[n] is a sorted list of (t, cx, cy) values for curve n,
        where *t* ∈ [0, 1] is the parameter along the segment, and
        (cx, cy) is the crossing point in data coords.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    K = U.shape[0]
    crossings = []

    for n in range(K):
        curve_y = U[n, :]
        hits = []
        for j in range(len(x) - 1):
            # Solve for t and s simultaneously:
            # x1 + t*dx = x[j] + s*(x[j+1]-x[j])
            # y1 + t*dy = curve_y[j] + s*(curve_y[j+1]-curve_y[j])
            ax = x[j]
            bx = x[j + 1] - x[j]
            ay = curve_y[j]
            by = curve_y[j + 1] - curve_y[j]

            denom = dx * by - dy * bx
            if abs(denom) < 1e-15:
                continue  # parallel

            t = ((ax - x1) * by - (ay - y1) * bx) / denom
            s = ((ax - x1) * dy - (ay - y1) * dx) / denom

            if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
                cx = x1 + t * dx
                cy = y1 + t * dy
                hits.append((t, cx, cy))

        hits.sort()
        crossings.append(hits)
    return crossings


# ── Interactive tool ─────────────────────────────────────────────────────

class ProfileTool:
    PICK_RADIUS = 8  # pixels

    def __init__(self, fig, ax_main, ax_profile, checkpoint: Checkpoint):
        self.fig = fig
        self.ax_main = ax_main
        self.ax_profile = ax_profile
        self.checkpoint = checkpoint

        self.U_norm = checkpoint.U - np.min(checkpoint.U)
        self.x = checkpoint.X

        # State: the two endpoints (None until placed)
        self.pts = [None, None]  # each is (x, y) or None
        self.placing_index = 0  # next point to place (0 or 1)

        # Artists on ax_main
        self.point_artists = [None, None]
        self.line_artist: Line2D | None = None

        # Dragging state
        self._dragging = None  # index of endpoint being dragged
        self._dragging_line = False  # True when dragging the whole segment
        self._drag_offset = None  # offsets at grab time

        # Draw the publication-style plot
        self._draw_main()

        # Connect events
        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _draw_main(self):
        ax = self.ax_main
        U = self.U_norm
        x = self.x

        for i in range(self.checkpoint.K):
            ax.plot(x, U[i, :], color="black", linewidth=0.7)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$u_n(x)$")
        ax.set_title("Click two points to define a profile line")

    def _set_point(self, idx, xy):
        self.pts[idx] = xy
        if self.point_artists[idx] is not None:
            self.point_artists[idx].remove()

        artist, = self.ax_main.plot(
            xy[0], xy[1], "o", color="red", markersize=8, picker=True, zorder=10,
        )
        self.point_artists[idx] = artist

    def _redraw_line(self):
        if self.line_artist is not None:
            self.line_artist.remove()
            self.line_artist = None

        if self.pts[0] is not None and self.pts[1] is not None:
            xs = [self.pts[0][0], self.pts[1][0]]
            ys = [self.pts[0][1], self.pts[1][1]]
            line, = self.ax_main.plot(xs, ys, "r-", linewidth=1.2, zorder=5)
            self.line_artist = line

    def _update_profile(self):
        ax = self.ax_profile
        ax.cla()

        if self.pts[0] is None or self.pts[1] is None:
            ax.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        ax.set_visible(True)
        p1, p2 = self.pts
        crossings = _intersections_along_line(self.x, self.U_norm, p1, p2)
        line_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        # Collect every crossing as (t, n, cx, cy), sorted by position along line
        all_hits: list[tuple[float, int, float, float]] = []
        for n, hits in enumerate(crossings):
            for t, cx, cy in hits:
                all_hits.append((t, n, cx, cy))
        all_hits.sort()

        if len(all_hits) < 2:
            ax.text(
                0.5, 0.5,
                "Line does not cross\nat least 2 curves",
                transform=ax.transAxes,
                ha="center", va="center",
            )
            self.fig.canvas.draw_idle()
            return

        # arc-length position of each crossing along the line
        arc = np.array([h[0] for h in all_hits]) * line_len

        # x-axis: 1-based crossing index
        # y-axis: gap distance to the previous crossing
        gap_n = np.arange(2, len(all_hits) + 1)
        gap_arc = np.diff(arc)
        gap_curve_idx = np.array([all_hits[i][1] for i in range(1, len(all_hits))])

        # ── main scatter + connecting line ──
        colors = plt.cm.tab10(np.linspace(0, 1, self.checkpoint.K))
        point_colors = [colors[ci % len(colors)] for ci in gap_curve_idx]

        ax.plot(gap_n, gap_arc, color="black", linewidth=0.6, zorder=3)
        ax.scatter(gap_n, gap_arc, c=point_colors, s=50, zorder=5,
                   edgecolors="black", linewidths=0.5)

        # ── log table ──
        dist_text_lines = []
        for i, (ni, d) in enumerate(zip(gap_n, gap_arc)):
            prev_curve = all_hits[i][1] + 1
            this_curve = all_hits[i + 1][1] + 1
            dist_text_lines.append(f"  crossing {ni - 1}→{ni}  (u_{prev_curve}→u_{this_curve}): {d:.6g}")

        # ── axes styling ──
        ax.set_xlabel("$n$")
        ax.set_ylabel(r"$\Delta U_n = u_{n} - u_{n-1}$")
        ax.set_title("Distance to previous crossing")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
        ax.margins(x=0.15, y=0.15)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

        logger.info(
            "Gap distances between successive crossings along the line:\n"
            + "\n".join(dist_text_lines)
        )
        self.fig.canvas.draw_idle()

    # ── event handlers ──────────────────────────────────────────────────

    def _pick_index(self, event):
        """Return index of the endpoint near *event* (pixel space), or None."""
        if event.inaxes != self.ax_main:
            return None
        ex, ey = self.ax_main.transData.transform((event.xdata, event.ydata))
        for idx in range(2):
            if self.pts[idx] is None:
                continue
            px, py = self.ax_main.transData.transform(self.pts[idx])
            if math.hypot(ex - px, ey - py) < self.PICK_RADIUS:
                return idx
        return None

    def _is_on_segment(self, event) -> bool:
        """Return True when *event* is within PICK_RADIUS pixels of the segment."""
        if self.pts[0] is None or self.pts[1] is None:
            return False
        if event.inaxes != self.ax_main:
            return False
        ex, ey = self.ax_main.transData.transform((event.xdata, event.ydata))
        ax0, ay0 = self.ax_main.transData.transform(self.pts[0])
        ax1, ay1 = self.ax_main.transData.transform(self.pts[1])

        seg_dx, seg_dy = ax1 - ax0, ay1 - ay0
        seg_len2 = seg_dx ** 2 + seg_dy ** 2
        if seg_len2 < 1e-10:
            return False

        t = ((ex - ax0) * seg_dx + (ey - ay0) * seg_dy) / seg_len2
        t = max(0.0, min(1.0, t))
        closest_x = ax0 + t * seg_dx
        closest_y = ay0 + t * seg_dy
        return math.hypot(ex - closest_x, ey - closest_y) < self.PICK_RADIUS

    def _on_press(self, event):
        if event.inaxes != self.ax_main or event.button != 1:
            return

        # 1. Try to pick an endpoint for dragging
        idx = self._pick_index(event)
        if idx is not None:
            self._dragging = idx
            return

        # 2. Try to grab the whole line segment
        if self._is_on_segment(event):
            self._dragging_line = True
            mx, my = event.xdata, event.ydata
            self._drag_offset = (
                (self.pts[0][0] - mx, self.pts[0][1] - my),
                (self.pts[1][0] - mx, self.pts[1][1] - my),
            )
            return

        # 3. Otherwise, place next point
        if self.placing_index <= 1:
            self._set_point(self.placing_index, (event.xdata, event.ydata))
            self.placing_index += 1
            if self.placing_index >= 2:
                self._redraw_line()
                self._update_profile()
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._dragging = None
        self._dragging_line = False
        self._drag_offset = None

    def _on_motion(self, event):
        if event.inaxes != self.ax_main:
            return
        if self._dragging is not None:
            idx = self._dragging
            self._set_point(idx, (event.xdata, event.ydata))
            self._redraw_line()
            self._update_profile()
        elif self._dragging_line and self._drag_offset is not None:
            mx, my = event.xdata, event.ydata
            self._set_point(0, (mx + self._drag_offset[0][0], my + self._drag_offset[0][1]))
            self._set_point(1, (mx + self._drag_offset[1][0], my + self._drag_offset[1][1]))
            self._redraw_line()
            self._update_profile()


# ── entry point ──────────────────────────────────────────────────────────

def main():
    cli = CLI().parse_args()

    if not cli.input.is_file():
        raise FileNotFoundError(f"{cli.input} is not a file.")

    checkpoint = Checkpoint.load_from_file(cli.input)
    logger.info(
        f"Loaded checkpoint: K={checkpoint.K}, M={checkpoint.M}, "
        f"L={checkpoint.L}, iter={checkpoint.iter}"
    )

    fig, (ax_main, ax_profile) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    ax_profile.set_visible(False)
    fig.subplots_adjust(wspace=0.35)

    _tool = ProfileTool(fig, ax_main, ax_profile, checkpoint)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()