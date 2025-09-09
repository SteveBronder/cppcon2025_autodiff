# newton_newton.py
# -*- coding: utf-8 -*-
"""Newton's method animation using Manim.

Fix:
- Exactly one per-step equation is shown at a time.
- After it reduces to x_{n+1} = value, it is immediately FadeOut + remove().
- Equation text is small and anchored next to the dot.
- Equation transitions are broken into stages:
  1. Show general Newton's formula.
  2. Substitute x_n with its numeric value.
  3. Substitute f(x_n) with its numeric value.
  4. Draw the tangent line.
  5. Substitute f'(x_n) with its numeric value (fully numeric fraction).
  6. Simplify to the next value x_{n+1}.
  7. Move the dot to the new x_{n+1}.
Each stage has a slight pause before proceeding to the next.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from manim import (
    Axes,
    BLUE_C,
    Create,
    DashedLine,
    DecimalNumber,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GREEN_C,
    GREY_C,
    LEFT,
    MathTex,
    RIGHT,
    Scene,
    TransformMatchingTex,
    UL,
    UP,
    UR,
    MovingCameraScene,
    ValueTracker,
    VGroup,
    WHITE,
    YELLOW_E,
    Write,
    always_redraw,
    config,
)

# -------------------------
# Global config & constants
# -------------------------

config.background_color = "#0e1117"  # dark

CURVE_COLOR = BLUE_C
TANGENT_COLOR = GREEN_C
GUIDE_COLOR = YELLOW_E
POINT_COLOR = WHITE
TRACE_COLOR = GREY_C
AXIS_COLOR = WHITE

CURVE_WIDTH = 3
TANGENT_WIDTH = 2
GUIDE_WIDTH = 2
AXIS_WIDTH = 1

DOT_RADIUS = 0.06
TRACE_RADIUS = 0.035

@dataclass(frozen=True)
class NewtonSceneSettings:
    """Immutable settings to parameterize the scene."""
    f: Callable[[float], float]
    fp: Callable[[float], float]
    stick_threshold: float = 0.06
    x0: float = 0.2886
    max_steps: int = 5
    # Axis ranges (min, max, step).
    x_range: Tuple[float, float, float] = (-0.5, 1.0, 0.25)
    y_range: Tuple[float, float, float] = (-1.0, 1.0, 0.25)

    # Equation sizing/placement.
    eq_scale: float = 0.30   # smaller so it hugs the dot
    eq_buff: float = 0.07    # distance from the dot

    # HUD sizing.
    hud_scale: float = 0.56
    # Timing controls for each animation step (all multiplied by t_scale).
    t_scale: float = 2.5
    # How long to draw the dashed guide lines (horizontal + vertical through the point).
    t_guides: float = 0.14
    # How long to show the general Newton formula when it first appears.
    t_eq_general: float = 0.70 * t_scale
    # Duration for each partial substitution transform (x_n, f(x_n), f'(x_n)).
    t_eq_sub_part: float = 0.40 * t_scale
    # Duration to reduce the substituted formula to the final simplified value.
    t_eq_reduce: float = 0.50 * t_scale
    # How long to draw the tangent line at the current point.
    t_tangent: float = 0.42 * t_scale
    # How long the "ghost" point slides along the tangent to the x-axis intercept.
    t_hop: float = 0.40 * t_scale
    # How long to draw the vertical drop from the intercept to the curve and snap the main dot to new position.
    t_drop_and_snap: float = 0.20 * t_scale
    # Hold time for fully substituted numeric equation (before final simplification).
    t_eq_hold: float = 0.80 * t_scale
    # Short pause between each substitution/tangent step.
    t_eq_pause: float = 0.10 * t_scale
    # Fade-out duration for the equation after showing x_{n+1} value.
    t_eq_fade: float = 0.40 * t_scale
    zoom_threshold: float = 0.07       # trigger zoom when |x| <= this
    zoom_target_scale: float = 0.30    # relative to initial frame width (smaller => closer)
    zoom_duration: float = 0.60 * t_scale
    # Equation sizing when zoomed-in (smaller + a touch tighter)
    eq_scale_zoomed: float = 0.08
    eq_buff_zoomed: float  = 0.05
    # Visual stroke widths when zoomed
    curve_width_zoom: float = 1.2
    axis_width_zoom: float = 0.8
    tangent_width_zoom: float = 1.0
    guide_width_zoom: float = 1.0
    dot_radius_zoom: float = 0.06
    trace_radius_zoom: float = 0.017

# -------------------------
# Math
# -------------------------

def f(x: float) -> float:
    """Function to root-find (your current choice)."""
    return x**3 - x**2 + 0.5 * x

def fp(x: float) -> float:
    """Analytical derivative."""
    return 3 * x**2 - 2 * x + 0.5

def newton_step(x: float, f_fn: Callable[[float], float], fp_fn: Callable[[float], float]) -> float:
    """One Newton update."""
    return x - (f_fn(x) / fp_fn(x))

def newton_iterates(x0: float, steps: int, f_fn: Callable[[float], float], fp_fn: Callable[[float], float]) -> List[float]:
    """Compute [x0, x1, ..., x_steps]."""
    xs: List[float] = [x0]
    x = x0
    for _ in range(steps):
        x = newton_step(x, f_fn, fp_fn)
        xs.append(x)
    return xs

# -------------------------
# Drawing helpers
# -------------------------

def make_tangent(ax: Axes, x: float, f_fn: Callable[[float], float],
                 fp_fn: Callable[[float], float], width: float):
    """Return a tangent line line Mobject at point (x, f(x)) on the curve."""
    y = f_fn(x); m = fp_fn(x)
    x_left, x_right = ax.x_range[0], ax.x_range[1]
    tan = ax.plot(lambda t: y + m * (t - x),
                  x_range=[x_left, x_right],
                  color=TANGENT_COLOR,
                  stroke_width=width)
    tan.set_z_index(3)
    return tan

def make_guides(ax: Axes, x: float, f_fn: Callable[[float], float], width: float) -> VGroup:
    """Return dashed horizontal and vertical guide lines through point (x, f(x))."""
    y = f_fn(x)
    horiz = DashedLine(ax.c2p(0.0, y), ax.c2p(x, y),
                       color=GUIDE_COLOR, stroke_width=width)
    vert  = DashedLine(ax.c2p(x, 0.0), ax.c2p(x, y),
                       color=GUIDE_COLOR, stroke_width=width)
    return VGroup(horiz, vert).set_z_index(2)

def equation_general(scale: float) -> MathTex:
    """General Newton update formula as MathTex."""
    return MathTex(
        r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}",
        substrings_to_isolate=[r"x_{n+1}"],
    ).scale(scale).set_z_index(6)

def equation_substituted(n: int, x_n: float, y_n: float, m: float, scale: float) -> MathTex:
    """Newton formula with numeric substitution for x_n, f(x_n), f'(x_n) (not simplified)."""
    return MathTex(
        rf"x_{{{n+1}}} = {x_n:.4f} - \frac{{{y_n:.4f}}}{{{m:.4f}}}",
        substrings_to_isolate=[rf"x_{{{n+1}}}"],
    ).scale(scale).set_z_index(6)

def equation_reduced(n: int, x_np1: float, scale: float) -> MathTex:
    """Final simplified result of Newton update (x_{n+1} = value)."""
    return MathTex(
        rf"x_{{{n+1}}} = {x_np1:.4f}",
        substrings_to_isolate=[rf"x_{{{n+1}}}"],
    ).scale(scale).set_z_index(6)

class NewtonRootScene(MovingCameraScene):
    """Scene illustrating Newton's method with step-by-step equation updates."""
    def construct(self) -> None:
        s = NewtonSceneSettings(f=f, fp=fp)
        # Axes and curve
        ax = Axes(
            x_range=s.x_range,
            y_range=s.y_range,
            tips=False,
            axis_config={
                "stroke_color": AXIS_COLOR,
                "stroke_width": AXIS_WIDTH,
                "include_numbers": False,
                "include_ticks": True,
                "tick_size": 0.06,
            },
        )
        frame = self.camera.frame
        base_width = frame.get_width()
        sticky_corner = ax.c2p(-0.4, 0.1)
        labels = ax.get_axis_labels(MathTex("").scale(0.7))
        curve = ax.plot(s.f, color=CURVE_COLOR, stroke_width=CURVE_WIDTH).set_z_index(1)
        # Current stroke widths (adjusted if we zoom in later)
        cur_curve_w   = CURVE_WIDTH
        cur_axis_w    = AXIS_WIDTH
        cur_tangent_w = TANGENT_WIDTH
        cur_guide_w   = GUIDE_WIDTH
        cur_trace_r   = TRACE_RADIUS

        # Draw axes and curve
        self.play(Create(ax), Write(labels), Create(curve), run_time=0.8)

        # Pre-compute the Newton iterates x0...xN
        xs = newton_iterates(s.x0, s.max_steps, s.f, s.fp)
        x_tracker = ValueTracker(xs[0])
        y_tracker = ValueTracker(s.f(xs[0]))

        # HUD for current x and f(x) values
        x_label = MathTex("x =").scale(s.hud_scale)
        y_label = MathTex("f(x) =").scale(s.hud_scale)
        x_value = DecimalNumber(xs[0], num_decimal_places=6, include_sign=True).scale(s.hud_scale)
        y_value = DecimalNumber(s.f(xs[0]), num_decimal_places=6, include_sign=True).scale(s.hud_scale)
        x_value.add_updater(lambda m: m.set_value(x_tracker.get_value()))
        y_value.add_updater(lambda m: m.set_value(y_tracker.get_value()))
        x_row = VGroup(x_label, x_value).arrange(RIGHT, buff=0.16)
        y_row = VGroup(y_label, y_value).arrange(RIGHT, buff=0.16)
        hud = VGroup(x_row, y_row).arrange(DOWN, aligned_edge=RIGHT, buff=0.10)
        hud.to_corner(UR).shift(LEFT * 0.30 + DOWN * 0.25)
        hud.set_z_index(10)
        # Keep HUD in corner on camera movement (zoom)
        hud.add_updater(lambda m: m.to_corner(UR).shift(LEFT * 0.30 + DOWN * 0.25))
        self.add(hud)
        self.play(FadeIn(hud), run_time=0.25)

        # Initial point on curve
        p_dot = Dot(ax.c2p(xs[0], s.f(xs[0])), radius=DOT_RADIUS, color=POINT_COLOR).set_z_index(5)
        self.play(FadeIn(p_dot), run_time=0.2)

        eq_current: MathTex | None = None  # track current equation mobject on screen

        # Main Newton iteration loop
        zoomed = False
        for n in range(s.max_steps):
            x_n, x_np1 = xs[n], xs[n + 1]
            y_n = s.f(x_n)
            if y_n == 0.0:
                break  # found exact root, end early

            y_np1 = s.f(x_np1)
            m = s.fp(x_n)

            # Apply zoom on first near-zero if threshold met
            if (not zoomed) and (abs(x_n) <= s.zoom_threshold):
                desired_width = base_width * s.zoom_target_scale
                scale_factor  = desired_width / frame.get_width()
                # Scale dot such that it maintains constant visible size after zoom
                dot_scale = (s.dot_radius_zoom / DOT_RADIUS) * scale_factor
                self.play(
                    frame.animate.scale(scale_factor).move_to(ax.c2p(0.0, 0.0)),
                    curve.animate.set_stroke(width=s.curve_width_zoom),
                    ax.animate.set_stroke(width=s.axis_width_zoom),
                    p_dot.animate.scale(dot_scale),
                    run_time=s.zoom_duration,
                )
                zoomed = True
                # Update stroke widths/radii for new zoomed scale
                cur_tangent_w = s.tangent_width_zoom
                cur_guide_w   = s.guide_width_zoom
                cur_trace_r   = s.trace_radius_zoom

            # Remove any lingering equation from previous step
            if eq_current is not None:
                self.remove(eq_current)
                eq_current = None

            # Create guides (dashed lines) through current point
            guides = make_guides(ax, x_n, s.f, cur_guide_w)
            # Determine if equation should stick in corner (if point is near y-axis)
            stick = abs(x_n) <= s.stick_threshold
            # Equation placement parameters
            eq_hdir = RIGHT + 3   # horizontal direction to offset near the dot
            eq_scale_now = s.eq_scale_zoomed if zoomed else s.eq_scale
            eq_buff_now  = s.eq_buff_zoomed  if zoomed else s.eq_buff

            def place_eq(mob: MathTex) -> MathTex:
                """Position equation MathTex either near the dot or in a fixed corner."""
                if stick:
                    # If too close to axis, place equation at a fixed location to avoid overlap
                    return mob.move_to(sticky_corner, aligned_edge=UL)
                # Otherwise, position it just to the right of the dot
                return mob.next_to(p_dot, eq_hdir, buff=eq_buff_now).match_y(p_dot)

            # **Step 1:** Show general Newton formula next to current point
            eq_gen = place_eq(equation_general(eq_scale_now))
            self.play(Create(guides), FadeIn(eq_gen), run_time=s.t_guides + s.t_eq_general)

            # **Step 2:** Substitute x_n value into the formula (x_n -> numeric value)
            eq_x = place_eq(MathTex(
                rf"x_{{{n+1}}} = {x_n:.4f} - \frac{{f({x_n:.4f})}}{{f'({x_n:.4f})}}",
                substrings_to_isolate=[rf"x_{{{n+1}}}"]
            ).scale(eq_scale_now).set_z_index(6))
            self.play(TransformMatchingTex(eq_gen, eq_x, transform_mismatches=True), run_time=s.t_eq_sub_part)
            self.remove(eq_gen)
            eq_current = eq_x
            self.wait(s.t_eq_pause)

            # **Step 3:** Substitute f(x_n) with its numeric value in the formula
            eq_x_f = place_eq(MathTex(
                rf"x_{{{n+1}}} = {x_n:.4f} - \frac{{{y_n:.4f}}}{{f'({x_n:.4f})}}",
                substrings_to_isolate=[rf"x_{{{n+1}}}"]
            ).scale(eq_scale_now).set_z_index(6))
            self.play(TransformMatchingTex(eq_x, eq_x_f, transform_mismatches=True), run_time=s.t_eq_sub_part)
            self.remove(eq_x)
            eq_current = eq_x_f
            self.wait(s.t_eq_pause)

            # **Step 4:** Draw the tangent line at the current point
            tangent = make_tangent(ax, x_n, s.f, s.fp, cur_tangent_w)
            self.play(Create(tangent), run_time=s.t_tangent)
            self.wait(s.t_eq_pause)

            # **Step 5:** Substitute f'(x_n) with its numeric value (now the formula is fully numeric)
            eq_x_fp = place_eq(equation_substituted(n, x_n, y_n, m, eq_scale_now))
            self.play(TransformMatchingTex(eq_x_f, eq_x_fp, transform_mismatches=True), run_time=s.t_eq_sub_part)
            self.remove(eq_x_f)
            eq_current = eq_x_fp
            # Hold the fully substituted equation on screen briefly
            self.wait(s.t_eq_hold)

            # **Step 6:** Simplify to get the next value x_{n+1}
            eq_red = place_eq(equation_reduced(n, x_np1, eq_scale_now))
            self.play(TransformMatchingTex(eq_x_fp, eq_red, transform_mismatches=True), run_time=s.t_eq_reduce)
            self.remove(eq_x_fp)
            eq_current = eq_red

            # **Step 7:** Slide a ghost dot along the tangent to the x-axis (visual hop to root estimate)
            ghost = Dot(ax.c2p(x_n, y_n), radius=cur_trace_r, color=TANGENT_COLOR).set_z_index(3)
            self.add(ghost)
            self.play(ghost.animate.move_to(ax.c2p(x_np1, 0.0)), run_time=s.t_hop)
            # Fade out the equation once the ghost reaches the x-axis
            self.play(FadeOut(eq_red), run_time=s.t_eq_fade)
            self.remove(eq_red)
            eq_current = None

            # Draw a vertical drop from the tangent intercept to the curve, and move the main dot to the new point
            drop = DashedLine(ax.c2p(x_np1, 0.0), ax.c2p(x_np1, y_np1),
                              color=GUIDE_COLOR, stroke_width=GUIDE_WIDTH)
            self.play(Create(drop), run_time=0.10)
            # Mark the old x-position on the x-axis with a trace dot
            self.add(Dot(ax.c2p(x_n, 0.0), radius=cur_trace_r, color=TRACE_COLOR))
            # Animate the main point moving to the new (x_{n+1}, f(x_{n+1})) position, updating HUD trackers
            self.play(
                p_dot.animate.move_to(ax.c2p(x_np1, y_np1)),
                x_tracker.animate.set_value(x_np1),
                y_tracker.animate.set_value(y_np1),
                run_time=s.t_drop_and_snap,
            )
            # Mark the new x-position on the x-axis with a trace dot
            self.add(Dot(ax.c2p(x_n, 0.0), radius=cur_trace_r, color=TRACE_COLOR))
            # Remove all per-step auxiliary visuals (ghost dot, tangent line, guides, drop line)
            self.play(FadeOut(ghost), FadeOut(tangent), FadeOut(guides), FadeOut(drop), run_time=0.16)

        # Place a final trace dot at the last root estimate on the x-axis
        self.add(Dot(ax.c2p(xs[-1], 0.0), radius=TRACE_RADIUS, color=TRACE_COLOR))
        self.wait(0.1)
