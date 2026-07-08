"""Bewegungstests für UR5e.

Jeder Test: Start → Ziel (TCP + Orientation in Grad).
Der Runner setzt den Roboter zwischen Tests zurück,
fährt die Startposition via RRT an und testet mit dem
angegebenen Mode (linear / rrt).
"""

import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pybullet as pb
from ur5e_bullet import UR5Sim


# ─── Helper ───────────────────────────────────────────────────────────────────

def _(x, y, z, rx=0, ry=0, rz=0):
    """Position + Orientation (Grad) → (pos, ori_rad)."""
    return [x, y, z], [math.radians(rx), math.radians(ry), math.radians(rz)]


def _reset_to_default(sim):
    """Roboter in bekannte, kollisionsfreie Pose setzen."""
    home = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
    for j, v in zip(sim._joint_ids, home):
        pb.resetJointState(sim.ur5, j, v)


# ─── Testfälle ────────────────────────────────────────────────────────────────
#
# expected:
#   success      move_to() = True, TCP ≤ 5 mm vom target
#   fail         move_to() = False (Kollision / kein Pfad / unreachable)
#   unreachable  IK liefert None, move_to() = False
#
# start / target:  (pos, ori)  – None = Roboter in Nullstellung
# mode:            "linear" | "rrt"

TESTS = [

    # ── Linear: Erfolg ────────────────────────────────────────────────────────

    dict(name="Linear nah",
         start=_(0.0, 0.3, 0.4),
         target=_(0.03, 0.32, 0.38),
         mode="linear", expected="success"),

    dict(name="Linear seitlich",
         start=_(0.0, 0.3, 0.4),
         target=_(0.03, 0.27, 0.4),
         mode="linear", expected="success"),

    dict(name="Linear runter",
         start=_(0.0, 0.3, 0.4),
         target=_(0.02, 0.3, 0.35),
         mode="linear", expected="success"),

    # ── Linear: Fehlschlag ────────────────────────────────────────────────────

    dict(name="Linear – Kollision mit Tisch",
         start=_(0.0, 0.3, 0.4),
         target=_(0.1, 0.25, 0.05),
         mode="linear", expected="fail"),

    dict(name="Linear – kein Pfad",
         start=_(0.0, 0.3, 0.4),
         target=_(0.02, 0.45, 0.08),
         mode="linear", expected="fail"),

    # ── Linear: Unreachable (IK > 8 cm) ────────────────────────────────────────

    dict(name="Linear – unreachable (zu weit)",
         start=_(0.0, 0.3, 0.4),
         target=_(0.8, 0.0, 1.0),
         mode="linear", expected="unreachable"),

    dict(name="Linear – unreachable (weit oben)",
         start=_(0.0, 0.3, 0.4),
         target=_(0.5, 0.5, 0.9),
         mode="linear", expected="unreachable"),

    # ── RRT: Erfolg ───────────────────────────────────────────────────────────

    dict(name="RRT einfach",
         start=_(0.0, 0.3, 0.4),
         target=_(0.05, 0.25, 0.35),
         mode="rrt", expected="success"),

    dict(name="RRT andere Seite",
         start=_(0.0, 0.3, 0.4),
         target=_(-0.15, 0.2, 0.3),
         mode="rrt", expected="success"),

    dict(name="RRT seitlich",
         start=_(0.0, 0.3, 0.4),
         target=_(0.05, 0.35, 0.35),
         mode="rrt", expected="success"),

    dict(name="RRT aus Nullstellung",
         start=None,
         target=_(0.1, -0.2, 0.35),
         mode="rrt", expected="success"),

    # ── RRT: Fehlschlag ──────────────────────────────────────────────────────

    dict(name="RRT – Kollision am Ziel",
         start=_(0.0, 0.3, 0.4),
         target=_(0.2, 0.0, -0.15),
         mode="rrt", expected="fail"),

    # ── RRT: Unreachable ──────────────────────────────────────────────────────

    dict(name="RRT – unreachable",
         start=_(0.0, 0.3, 0.4),
         target=_(0.8, 0.0, 1.0),
         mode="rrt", expected="unreachable"),

]


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_test(test, sim):
    name = test["name"]
    mode = test["mode"]
    expected = test["expected"]
    target_pos, target_ori = test["target"]
    t0 = time.time()

    _reset_to_default(sim)
    start = test.get("start")
    if start is not None:
        start_pos, start_ori = start
        if not sim.move_to(start_pos, start_ori, linear=False):
            return False, "Setup fehlgeschlagen", time.time() - t0

    linear = (mode == "linear")
    ok = sim.move_to(target_pos, target_ori, linear=linear)
    elapsed = time.time() - t0

    tcp, _ = sim.get_tcp_pose()
    error_mm = math.sqrt(sum((tcp[i]-target_pos[i])**2 for i in range(3))) * 1000

    passed, detail = _check(expected, ok, error_mm)
    return passed, detail, elapsed


def _check(expected, ok, error_mm):
    if expected == "success":
        return (ok and error_mm <= 5), f"{error_mm:.1f} mm"
    if expected == "fail":
        return (not ok), f"move_to={ok}"
    if expected == "unreachable":
        return (not ok), f"move_to={ok}"
    return False, f"unbekannt: {expected}"


def run_tests(tests, gui=False):
    pb.connect(pb.GUI if gui else pb.DIRECT)
    sim = UR5Sim()

    print(f"\n{'─'*70}")
    print(f"  UR5e Bewegungstests — {len(tests)} Fälle")
    print(f"{'─'*70}")

    results = []
    for tc in tests:
        passed, detail, elapsed = run_test(tc, sim)
        name = tc["name"]
        status = "✓" if passed else "✗"
        print(f"  {status} {name:42s}  {detail:25s}  ({elapsed:.1f}s)")
        results.append((passed, name))

    pb.disconnect()

    passed_count = sum(1 for p, _ in results if p)
    print(f"{'─'*70}")
    print(f"  Ergebnis: {passed_count}/{len(results)} bestanden")
    print(f"{'─'*70}\n")
    return all(p for p, _ in results)


if __name__ == "__main__":
    run_tests(TESTS)
