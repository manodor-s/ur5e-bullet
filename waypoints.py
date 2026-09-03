import math

import pybullet as pb


def _quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q] if n > 0 else [1.0, 0.0, 0.0, 0.0]


def _quat_mul(a, b):
    """Quaternion-Produkt (Hamilton, xyzw wie pybullet): R(a)·R(b)."""
    a1, a2, a3, a0 = a
    b1, b2, b3, b0 = b
    return [
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
    ]


def _quat_slerp(a, b, t):
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = [-c for c in b]
        dot = -dot
    if dot > 0.9995:
        r = [a[i] + t * (b[i] - a[i]) for i in range(4)]
        return _quat_normalize(r)
    theta = math.acos(min(1.0, dot))
    so = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / so
    wb = math.sin(t * theta) / so
    return [wa * a[i] + wb * b[i] for i in range(4)]


def parabola_waypoints(cfg):
    p = cfg.get("parabola", {})
    n = int(p.get("n", 20))
    y_max = float(p.get("y_max", 0.05))
    x0 = float(p.get("x0", 0.0))
    a = float(p.get("a", 1.0))
    z = float(p.get("z", 0.0))
    power = float(p.get("power", 4.0))

    # Per-Startposition schaltbar: Kamera-Blickachse (TCP-lokales -Z) auf den
    # Look-Target-Punkt in der Hoehe der Waypoint-Ebene richten (Blickachse
    # horizontal in der Ebene). Ziel = look_target (x,y) oder Default Gebiss-Mitte.
    # Default an, kann je Startposition gesetzt werden.
    look_at_jaw = cfg.get("look_at_jaw", True)
    lt = cfg.get("look_target")
    jaw = cfg.get("jaw_pos")
    if lt is not None:
        target_xy = [lt[0], lt[1]]
    elif jaw is not None:
        target_xy = [jaw[0], jaw[1]]
    else:
        target_xy = None

    anchors = cfg.get("ori_anchors", {})
    a_start = anchors.get("start", [90, 0, 0])
    a_mid = anchors.get("mid", [180, 90, 0])
    a_end = anchors.get("end", [-90, 0, 0])
    q_start = pb.getQuaternionFromEuler([math.radians(v) for v in a_start])
    q_mid = pb.getQuaternionFromEuler([math.radians(v) for v in a_mid])
    q_end = pb.getQuaternionFromEuler([math.radians(v) for v in a_end])

    mid = n // 2
    last = n - 1

    # Bogenlaenge entlang der Kurve numerisch berechnen: v -> kumulierte Laenge.
    # x(v) = x0 + a*|v/y_max|^power, y(v) = v  -> ds/dv = sqrt((dx/dv)^2 + 1)
    def dx_dv(v):
        return a * power * abs(v / y_max) ** (power - 1) / y_max if y_max > 0 else 0.0

    n_int = 4000
    v_grid = []
    L_grid = []
    acc = 0.0
    prev = -y_max
    prev_x = dx_dv(prev)
    v_grid.append(prev)
    L_grid.append(0.0)
    for k in range(1, n_int + 1):
        v = -y_max + 2 * y_max * k / n_int
        d = dx_dv(v)
        acc += math.sqrt(((prev_x + d) / 2.0) ** 2 + 1.0) * (v - prev)
        prev = v
        prev_x = d
        v_grid.append(v)
        L_grid.append(acc)
    L_total = acc

    if n > 1 and L_total > 0:
        values = []
        for i in range(n):
            target = L_total * i / (n - 1)
            # invertiere L(v)=target per linearer Interpolation (L monoton)
            for k in range(1, len(L_grid)):
                if L_grid[k] >= target:
                    v0, v1 = v_grid[k - 1], v_grid[k]
                    l0, l1 = L_grid[k - 1], L_grid[k]
                    if l1 > l0:
                        f = (target - l0) / (l1 - l0)
                    else:
                        f = 0.0
                    values.append(v0 + f * (v1 - v0))
                    break
            else:
                values.append(v_grid[-1])
    else:
        values = [0.0] * n

    wps = []
    for idx in range(n):
        value = values[idx]
        x = x0 + a * abs(value / y_max) ** power if y_max > 0 else x0
        if idx <= mid:
            t = idx / mid if mid > 0 else 0.0
            q = _quat_slerp(q_start, q_mid, t)
        else:
            t = (idx - mid) / (last - mid) if last > mid else 1.0
            q = _quat_slerp(q_mid, q_end, t)

        if look_at_jaw and target_xy is not None:
            wp_pos = [x, value, z]
            d = [target_xy[0] - wp_pos[0], target_xy[1] - wp_pos[1], 0.0]
            dn = math.sqrt(sum(c * c for c in d))
            if dn > 1e-9:
                d = [c / dn for c in d]
                R = pb.getMatrixFromQuaternion(q)
                f0 = [-(R[2]), -(R[5]), -(R[8])]
                axis = [
                    f0[1]*d[2] - f0[2]*d[1],
                    f0[2]*d[0] - f0[0]*d[2],
                    f0[0]*d[1] - f0[1]*d[0],
                ]
                an = math.sqrt(sum(c*c for c in axis))
                if an > 1e-6:
                    axis = [c/an for c in axis]
                    dot = f0[0]*d[0] + f0[1]*d[1] + f0[2]*d[2]
                    ang = math.acos(max(-1.0, min(1.0, dot)))
                    dq = pb.getQuaternionFromAxisAngle(axis, ang)
                    q = _quat_mul(dq, q)

        rot = [math.degrees(v) for v in pb.getEulerFromQuaternion(q)]
        wps.append({
            "name": f"W{idx}",
            "value": value,
            "tcp_pos": [x, value, z],
            "tcp_ori_deg": rot,
        })
    return wps
