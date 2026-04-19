"""
=============================================================================
  3D Physics & Light Visualization Engine
  Linear Algebra Research Project
  ─────────────────────────────────────────────────────────────────────────
  Core LA concepts used throughout:
    • 4×4 Homogeneous Transformation Matrices  (translation, rotation, scale)
    • Matrix–Matrix Multiplication             (composing transforms)
    • Matrix–Vector Multiplication             (transforming vertices)
    • Perspective Projection Matrix            (3-D → 2-D)
    • Dot Product                              (lighting, normals, shadows)
    • Cross Product                            (surface-normal computation)
    • Vector Reflection Formula  R = D − 2(D·n̂)n̂
    • Shadow Projection onto a Ground Plane    (plane–ray intersection)
=============================================================================
"""

import pygame
import numpy as np
import sys
import math

# ─────────────────────────────────────────────────────────────────────────────
# WINDOW & SCENE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 1280, 800
UI_PANEL_WIDTH  = 300          # right-hand control panel
SCENE_WIDTH     = WIDTH - UI_PANEL_WIDTH
FOV             = 60           # horizontal field-of-view (degrees)
NEAR, FAR       = 0.1, 100.0  # clip planes
FPS             = 60

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR         = (12,  14,  20)
PANEL_COLOR      = (20,  24,  36)
PANEL_BORDER     = (40,  50,  80)
GRID_COLOR       = (30,  38,  60)
FLOOR_COLOR      = (22,  28,  45)
OBJECT_COLOR     = (70, 140, 220)
EDGE_COLOR       = (120, 190, 255)
SHADOW_COLOR     = (0,    0,   0)          # alpha applied at draw time
SUN_COLOR        = (255, 220,  80)
RAY_COLOR        = (255, 200,  60)
REFLECT_COLOR    = (100, 255, 180)
TEXT_COLOR       = (200, 210, 240)
ACCENT_COLOR     = (80,  160, 255)
SLIDER_BG        = (35,  42,  65)
SLIDER_FG        = (80,  140, 255)
SLIDER_HANDLE    = (160, 200, 255)


# =============================================================================
# ░░  LINEAR ALGEBRA UTILITIES  ░░
# =============================================================================

def identity() -> np.ndarray:
    """Return the 4×4 identity matrix."""
    return np.eye(4, dtype=np.float64)


def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    4×4 Homogeneous Translation Matrix.
    Moves a point P by vector (tx, ty, tz):
      T·[x, y, z, 1]ᵀ  =  [x+tx, y+ty, z+tz, 1]ᵀ
    """
    M = identity()
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    4×4 Homogeneous Scale Matrix.
    S·[x,y,z,1]ᵀ = [sx·x, sy·y, sz·z, 1]ᵀ
    """
    M = identity()
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def rotation_x(angle_rad: float) -> np.ndarray:
    """
    4×4 Rotation about the X-axis (right-hand rule).
    Uses the standard rotation submatrix:
      [1    0       0   ]
      [0   cos θ  -sin θ]
      [0   sin θ   cos θ]
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    M = identity()
    M[1, 1] =  c;  M[1, 2] = -s
    M[2, 1] =  s;  M[2, 2] =  c
    return M


def rotation_y(angle_rad: float) -> np.ndarray:
    """
    4×4 Rotation about the Y-axis.
      [ cos θ  0  sin θ]
      [  0     1   0   ]
      [-sin θ  0  cos θ]
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    M = identity()
    M[0, 0] =  c;  M[0, 2] =  s
    M[2, 0] = -s;  M[2, 2] =  c
    return M


def rotation_z(angle_rad: float) -> np.ndarray:
    """
    4×4 Rotation about the Z-axis.
      [cos θ  -sin θ  0]
      [sin θ   cos θ  0]
      [ 0       0     1]
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    M = identity()
    M[0, 0] =  c;  M[0, 1] = -s
    M[1, 0] =  s;  M[1, 1] =  c
    return M


def perspective_matrix(fov_deg: float, aspect: float,
                       near: float, far: float) -> np.ndarray:
    """
    Standard OpenGL-style Perspective Projection Matrix (column-major).
    Maps the view frustum to NDC (Normalised Device Coordinates).

    Key LA idea: the w-component after multiplication encodes depth;
    the final divide by w (perspective divide) performs the non-linear
    foreshortening that makes parallel lines converge.

        f = 1 / tan(fov/2)
        P = | f/a   0      0             0          |
            |  0    f      0             0          |
            |  0    0   (f+n)/(n-f)   2fn/(n-f)    |
            |  0    0     -1             0          |
    """
    fov_rad = math.radians(fov_deg)
    f       = 1.0 / math.tan(fov_rad / 2.0)
    M       = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M


def look_at(eye: np.ndarray, center: np.ndarray,
            up: np.ndarray) -> np.ndarray:
    """
    View Matrix via Gram–Schmidt orthonormalisation.
    Constructs an orthonormal camera basis {right, up', forward},
    then builds the view transform as  V = R · T(-eye).
    """
    # LA: normalise direction vectors (unit vectors)
    f = center - eye;   f /= np.linalg.norm(f)      # forward
    r = np.cross(f, up); r /= np.linalg.norm(r)     # right  (cross product)
    u = np.cross(r, f)                               # corrected up

    V = np.eye(4, dtype=np.float64)
    V[0, :3] = r
    V[1, :3] = u
    V[2, :3] = -f
    # Translate by -dot(basis, eye)  ← DOT PRODUCTS
    V[0, 3]  = -np.dot(r, eye)
    V[1, 3]  = -np.dot(u, eye)
    V[2, 3]  =  np.dot(f, eye)
    return V


def transform_vertices(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply a 4×4 matrix to an (N,3) array of 3-D vertices.
    Converts to homogeneous coordinates [x,y,z,1], multiplies,
    then strips w — this is a MATRIX–VECTOR MULTIPLICATION repeated N times.
    """
    N    = verts.shape[0]
    h    = np.ones((N, 4), dtype=np.float64)
    h[:, :3] = verts
    # LA: matrix–matrix product (broadcast over N rows)
    out  = (M @ h.T).T          # shape (N,4)
    return out[:, :3] / out[:, 3:4]   # perspective divide if needed


def project_to_screen(verts_3d: np.ndarray,
                      VP: np.ndarray,            # view–projection matrix
                      sw: int, sh: int) -> np.ndarray:
    """
    Full 3-D → 2-D pipeline:
      1. Multiply by the combined View–Projection matrix  (MATRIX MULTIPLICATION)
      2. Perspective divide: divide x,y,z by w           (PROJECTION)
      3. Viewport transform: NDC [-1,1] → pixel coords
    Returns (N, 2) array of screen pixels, plus (N,) depth array.
    """
    N    = verts_3d.shape[0]
    h    = np.ones((N, 4), dtype=np.float64)
    h[:, :3] = verts_3d

    # LA: single (4,4)·(4,N) matrix multiplication
    clip  = (VP @ h.T).T          # clip space, shape (N,4)
    w     = clip[:, 3:4]
    ndc   = clip[:, :3] / w       # perspective divide → NDC

    # Viewport transform (affine map)
    sx    =  (ndc[:, 0] + 1.0) * 0.5 * sw
    sy    = (-ndc[:, 1] + 1.0) * 0.5 * sh
    depth =   ndc[:, 2]
    return np.stack([sx, sy], axis=1), depth


def face_normal(v0, v1, v2):
    """
    Compute the unit surface normal of a triangular face via CROSS PRODUCT.
    n = (v1-v0) × (v2-v0),  then normalise.
    """
    a = v1 - v0
    b = v2 - v0
    n = np.cross(a, b)          # LA: cross product → perpendicular vector
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def reflect_ray(D: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Vector Reflection Formula:  R = D − 2(D·n̂)n̂
    ──────────────────────────────────────────────
    • D  : incident direction (unit vector pointing FROM light source)
    • n̂  : surface unit normal at the hit point
    • D·n̂: DOT PRODUCT — scalar projection of D onto n̂
    Returns the reflected direction vector R.
    """
    n_hat = n / np.linalg.norm(n)
    # LA: DOT PRODUCT used here
    dot   = np.dot(D, n_hat)
    R     = D - 2.0 * dot * n_hat
    return R / np.linalg.norm(R)


def shadow_project_point(P: np.ndarray,
                         light: np.ndarray,
                         ground_z: float = 0.0) -> np.ndarray:
    """
    Project point P onto the ground plane Z = ground_z
    along the ray from the light source.

    PLANE–RAY INTERSECTION (Linear Algebra):
      Ray:   Q(t) = light + t·(P − light)
      Plane: Z = ground_z  →  light_z + t·(P_z − light_z) = ground_z
      t = (ground_z − light_z) / (P_z − light_z)

    The result is an affine combination of the light and the point.
    """
    D   = P - light                        # ray direction vector
    dz  = D[2]
    if abs(dz) < 1e-6:
        return P.copy()
    t   = (ground_z - light[2]) / dz      # scalar parameter along ray
    S   = light + t * D                   # shadow point on ground plane
    return S


# =============================================================================
# ░░  GEOMETRY DEFINITIONS  ░░
# =============================================================================

def make_cube() -> tuple:
    """
    Unit cube centred at the origin.
    Returns (vertices[8×3], faces[list of vertex-index tuples]).
    """
    v = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
    ], dtype=np.float64)
    faces = [
        (0,1,2,3),  # bottom
        (4,5,6,7),  # top
        (0,1,5,4),  # front
        (2,3,7,6),  # back
        (0,3,7,4),  # left
        (1,2,6,5),  # right
    ]
    return v, faces


def make_pyramid() -> tuple:
    """Square-base pyramid."""
    v = np.array([
        [-0.5, -0.5, -0.5],  # 0 base corners
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [ 0.0,  0.0,  0.7],  # 4 apex
    ], dtype=np.float64)
    faces = [
        (0,1,2,3),  # base
        (0,1,4),
        (1,2,4),
        (2,3,4),
        (3,0,4),
    ]
    return v, faces


def make_prism(sides: int = 6) -> tuple:
    """
    Regular N-sided prism (default: hexagonal).
    Vertices are generated with polar coordinates — a nice use of
    trigonometric parametric equations.
    """
    verts  = []
    r      = 0.55
    h      = 0.5
    angles = [2 * math.pi * i / sides for i in range(sides)]

    # Bottom ring
    for a in angles:
        verts.append([r * math.cos(a), r * math.sin(a), -h])
    # Top ring
    for a in angles:
        verts.append([r * math.cos(a), r * math.sin(a),  h])

    v  = np.array(verts, dtype=np.float64)
    faces = []

    # Side quads
    for i in range(sides):
        j  = (i + 1) % sides
        faces.append((i, j, j + sides, i + sides))

    # Top and bottom caps (as one polygon)
    faces.append(tuple(range(sides)))              # bottom cap
    faces.append(tuple(range(sides, 2 * sides)))   # top cap

    return v, faces


# =============================================================================
# ░░  SLIDER UI WIDGET  ░░
# =============================================================================

class Slider:
    """A simple horizontal slider for the control panel."""
    def __init__(self, label: str, x: int, y: int, w: int,
                 vmin: float, vmax: float, value: float,
                 color=SLIDER_FG):
        self.label  = label
        self.rect   = pygame.Rect(x, y, w, 6)
        self.vmin   = vmin
        self.vmax   = vmax
        self.value  = value
        self.color  = color
        self.drag   = False
        self.handle_r = 8

    @property
    def t(self):
        """Normalised position in [0,1]."""
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    @property
    def handle_x(self):
        return int(self.rect.x + self.t * self.rect.w)

    def draw(self, surf: pygame.Surface, font):
        # Track
        pygame.draw.rect(surf, SLIDER_BG, self.rect, border_radius=3)
        # Fill
        fill = pygame.Rect(self.rect.x, self.rect.y,
                           int(self.t * self.rect.w), self.rect.h)
        pygame.draw.rect(surf, self.color, fill, border_radius=3)
        # Handle
        hx = self.handle_x
        hy = self.rect.centery
        pygame.draw.circle(surf, SLIDER_HANDLE, (hx, hy), self.handle_r)
        # Label + value
        label_surf = font.render(f"{self.label}: {self.value:.2f}",
                                 True, TEXT_COLOR)
        surf.blit(label_surf, (self.rect.x, self.rect.y - 18))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            hx = self.handle_x
            hy = self.rect.centery
            dist = math.hypot(event.pos[0] - hx, event.pos[1] - hy)
            if dist <= self.handle_r + 4:
                self.drag = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        elif event.type == pygame.MOUSEMOTION and self.drag:
            rx = event.pos[0] - self.rect.x
            t  = max(0.0, min(1.0, rx / self.rect.w))
            self.value = self.vmin + t * (self.vmax - self.vmin)


class Button:
    """Simple toggle/press button."""
    def __init__(self, label, x, y, w, h, color=ACCENT_COLOR):
        self.label   = label
        self.rect    = pygame.Rect(x, y, w, h)
        self.color   = color
        self.pressed = False

    def draw(self, surf, font):
        c = tuple(min(255, int(v * 1.3)) for v in self.color) if self.pressed else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=5)
        pygame.draw.rect(surf, PANEL_BORDER, self.rect, 1, border_radius=5)
        t = font.render(self.label, True, (240, 248, 255))
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle_event(self, event) -> bool:
        """Returns True on click."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False
        return False


# =============================================================================
# ░░  MAIN ENGINE CLASS  ░░
# =============================================================================

class Engine3D:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3D Physics & Light Visualization Engine")
        self.clock   = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        # ── Scene State ──────────────────────────────────────────────────────
        self.auto_rotate  = True
        self.rot_y        = 0.0       # auto-rotation angle (radians)
        self.rot_x        = 0.3       # tilt for better perspective

        self.shape_names  = ["Cube", "Pyramid", "Prism"]
        self.shape_idx    = 0
        self.verts, self.faces = make_cube()

        # Camera (fixed at a comfortable distance)
        self.cam_eye    = np.array([0.0, -4.0, 2.5])
        self.cam_center = np.array([0.0,  0.0, 0.5])
        self.cam_up     = np.array([0.0,  0.0, 1.0])

        # ── Projection ───────────────────────────────────────────────────────
        aspect          = SCENE_WIDTH / HEIGHT
        self.P_mat      = perspective_matrix(FOV, aspect, NEAR, FAR)
        self.V_mat      = look_at(self.cam_eye, self.cam_center, self.cam_up)
        # Combined View–Projection matrix (MATRIX MULTIPLICATION)
        self.VP_mat     = self.P_mat @ self.V_mat

        # ── Control Panel Setup ───────────────────────────────────────────
        px   = SCENE_WIDTH + 16
        pw   = UI_PANEL_WIDTH - 28
        # Each slider row = 16px label + 6px track + 8px bottom padding = 30px
        # Section header rows add an extra 24px gap before them.
        ITEM_H    = 44   # pixels per slider row (label + track + breathing room)
        SEC_GAP   = 10   # extra pixels before each section header

        # Layout cursor — starts below the panel title
        cy = 42

        # ── Object section ────────────────────────────────────────────────
        # "── Object ──" header drawn at cy; sliders start below it
        self._sec_object_y = cy;  cy += 40   # header(~16px) + label gap(18px) + 6px spare
        self.sliders = {}

        for key, label, vmin, vmax, val, kw in [
            ("obj_x",     "Obj X",   -3,   3,   0.0, {}),
            ("obj_y",     "Obj Y",   -3,   3,   0.0, {}),
            ("obj_z",     "Obj Z",    0,   3,   0.5, {}),
            ("obj_scale", "Scale",  0.2,   3,   1.0, {}),
        ]:
            self.sliders[key] = Slider(label, px, cy, pw, vmin, vmax, val, **kw)
            cy += ITEM_H

        # ── Sun section ───────────────────────────────────────────────────
        cy += SEC_GAP
        self._sec_sun_y = cy;  cy += 40   # header + label gap + spare
        for key, label, vmin, vmax, val in [
            ("sun_x",     "Sun X",   -8,  8,   3.0),
            ("sun_y",     "Sun Y",   -8,  8,  -3.0),
            ("sun_z",     "Sun Z",    1, 12,   6.0),
            ("intensity", "Intensity", 0,  1,  0.6),
        ]:
            color = SUN_COLOR if key.startswith("sun") else SLIDER_FG
            self.sliders[key] = Slider(label, px, cy, pw, vmin, vmax, val, color)
            cy += ITEM_H

        # ── Shapes section ────────────────────────────────────────────────
        cy += SEC_GAP
        self._sec_shapes_y = cy;  cy += 26
        bw = (pw - 10) // 3
        self.btn_cube    = Button("Cube",        px,            cy, bw, 30)
        self.btn_pyramid = Button("Pyramid",     px + bw + 5,   cy, bw, 30)
        self.btn_prism   = Button("Prism",       px + 2*(bw+5), cy, bw, 30)
        cy += 40
        self.btn_rotate  = Button("Auto-Rot ON", px,            cy, pw, 30, (60, 120, 80))

        self.show_shadow  = True
        self.show_reflect = True

        # Off-screen surface for transparent shadow blending
        self.shadow_surf = pygame.Surface((SCENE_WIDTH, HEIGHT), pygame.SRCALPHA)

    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRY HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model_matrix(self) -> np.ndarray:
        """
        Compose the model's Transform matrix:
          M = T · Ry · Rx · S
        This is a chain of MATRIX MULTIPLICATIONS (non-commutative!).
        Order matters — we scale first, then rotate, then translate.
        """
        s  = self.sliders["obj_scale"].value
        S  = scale_matrix(s, s, s)
        Rx = rotation_x(self.rot_x)
        Ry = rotation_y(self.rot_y)
        T  = translation_matrix(
                self.sliders["obj_x"].value,
                self.sliders["obj_y"].value,
                self.sliders["obj_z"].value)
        # LA: chain of 4×4 matrix multiplications
        return T @ Ry @ Rx @ S

    def _world_vertices(self) -> np.ndarray:
        """
        Apply model matrix to canonical vertices → world-space positions.
        This is the MODEL TRANSFORM step of the MVP pipeline.
        """
        M   = self._build_model_matrix()
        # Homogeneous multiplication: vertices go from local → world space
        return transform_vertices(self.verts, M)

    def _sun_position(self) -> np.ndarray:
        return np.array([
            self.sliders["sun_x"].value,
            self.sliders["sun_y"].value,
            self.sliders["sun_z"].value,
        ], dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # DRAWING ROUTINES
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_floor(self, surf: pygame.Surface):
        """
        Draw a perspective-projected grid on the ground plane Z = 0.
        Each grid vertex is passed through the View–Projection matrix.
        """
        extent = 4
        step   = 1
        # Gather all grid intersection points
        pts = []
        for x in range(-extent, extent + 1, step):
            for y in range(-extent, extent + 1, step):
                pts.append([x, y, 0.0])
        pts = np.array(pts, dtype=np.float64)

        if len(pts) == 0:
            return

        # LA: PROJECT all floor points in one batch matrix multiplication
        screen_pts, depths = project_to_screen(pts, self.VP_mat, SCENE_WIDTH, HEIGHT)

        # Convert to a lookup dict keyed by (ix, iy)
        idx = 0
        grid = {}
        for xi in range(-extent, extent + 1, step):
            for yi in range(-extent, extent + 1, step):
                d = depths[idx]
                if -1.0 < d < 1.0:
                    grid[(xi, yi)] = screen_pts[idx].astype(int)
                else:
                    grid[(xi, yi)] = None
                idx += 1

        # Draw horizontal and vertical grid lines
        for xi in range(-extent, extent, step):
            for yi in range(-extent, extent, step):
                a = grid.get((xi,   yi))
                b = grid.get((xi+1, yi))
                c = grid.get((xi,   yi+1))
                if a is not None and b is not None:
                    pygame.draw.line(surf, GRID_COLOR, a, b, 1)
                if a is not None and c is not None:
                    pygame.draw.line(surf, GRID_COLOR, a, c, 1)

    def _draw_shadow(self, world_verts: np.ndarray, surf: pygame.Surface):
        """
        Project each vertex onto the ground plane (Z=0) along the ray from
        the light source (PLANE–RAY INTERSECTION).
        Then draw the shadow polygon using the projected 2-D points.
        """
        sun    = self._sun_position()
        alpha  = int(self.sliders["intensity"].value * 180)
        shadow_color = (0, 0, 0, alpha)

        # For each face, compute shadow quad on the ground plane
        shadow_polys = []
        for face in self.faces:
            shadow_face_verts = []
            for i in face:
                wv  = world_verts[i]
                # LA: PLANE–RAY INTERSECTION to find shadow point
                sv  = shadow_project_point(wv, sun, ground_z=0.0)
                shadow_face_verts.append(sv)

            pts = np.array(shadow_face_verts, dtype=np.float64)
            # LA: PROJECT shadow vertices through VP matrix
            sp, depths = project_to_screen(pts, self.VP_mat, SCENE_WIDTH, HEIGHT)

            # Only draw faces whose shadow vertices are on-screen
            valid = np.all((depths > -1.0) & (depths < 1.0))
            if valid and len(sp) >= 3:
                poly = [tuple(p.astype(int)) for p in sp]
                shadow_polys.append(poly)

        self.shadow_surf.fill((0, 0, 0, 0))
        for poly in shadow_polys:
            if len(poly) >= 3:
                pygame.draw.polygon(self.shadow_surf, shadow_color, poly)
        surf.blit(self.shadow_surf, (0, 0))

    def _draw_object(self, world_verts: np.ndarray, surf: pygame.Surface):
        """
        Render the 3-D object with flat shading.
        Lighting computed via DOT PRODUCT between face normal and light dir.
        Faces sorted back-to-front (Painter's Algorithm).
        """
        sun      = self._sun_position()
        cam_fwd  = self.cam_center - self.cam_eye
        cam_fwd /= np.linalg.norm(cam_fwd)

        # Project all world vertices to screen in ONE matrix batch
        # LA: batch MATRIX–VECTOR MULTIPLICATION
        sp, depths = project_to_screen(world_verts, self.VP_mat, SCENE_WIDTH, HEIGHT)

        face_data = []
        for face in self.faces:
            vx = world_verts[face[0]]
            vy = world_verts[face[1]]
            vz = world_verts[face[2]]

            # LA: CROSS PRODUCT to get surface normal
            n = face_normal(vx, vy, vz)

            # Back-face culling: skip faces pointing away from the camera
            to_cam = self.cam_eye - vx
            if np.dot(n, to_cam) < 0:    # LA: DOT PRODUCT for culling test
                continue

            # Phong-style diffuse lighting: intensity = max(0, n · L̂)
            L = sun - vx                  # vector from face to light
            L_hat = L / (np.linalg.norm(L) + 1e-10)
            # LA: DOT PRODUCT for Lambertian (diffuse) shading
            diffuse = max(0.05, np.dot(n, L_hat))

            # Average depth of face for Painter's sort
            avg_depth = np.mean([depths[i] for i in face])

            poly_screen = [tuple(sp[i].astype(int)) for i in face]
            face_data.append((avg_depth, poly_screen, diffuse, n, vx))

        # Sort faces back-to-front (Painter's Algorithm)
        face_data.sort(key=lambda x: x[0], reverse=True)

        for avg_depth, poly, diffuse, n, vx in face_data:
            if len(poly) < 3:
                continue
            r = int(OBJECT_COLOR[0] * diffuse)
            g = int(OBJECT_COLOR[1] * diffuse)
            b = int(OBJECT_COLOR[2] * diffuse)
            color = (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))
            pygame.draw.polygon(surf, color, poly)
            pygame.draw.polygon(surf, EDGE_COLOR, poly, 1)

    def _draw_sun_and_rays(self, world_verts: np.ndarray, surf: pygame.Surface):
        """
        Draw:
         1. The sun sphere (projected from 3-D world space)
         2. Incident ray from sun to object centre
         3. Reflected ray using R = D − 2(D·n̂)n̂
        """
        sun = self._sun_position()

        # Object centre (mean of world vertices)
        centre = np.mean(world_verts, axis=0)

        # ── Project sun & centre ──────────────────────────────────────────
        pts    = np.array([sun, centre], dtype=np.float64)
        sp, dp = project_to_screen(pts, self.VP_mat, SCENE_WIDTH, HEIGHT)

        sun_screen    = tuple(sp[0].astype(int))
        centre_screen = tuple(sp[1].astype(int))

        on_screen = lambda p, d: (0 <= p[0] < SCENE_WIDTH and
                                   0 <= p[1] < HEIGHT and -1 < d < 1)

        # Sun circle
        if on_screen(sun_screen, dp[0]):
            pygame.draw.circle(surf, SUN_COLOR, sun_screen, 14)
            pygame.draw.circle(surf, (255, 255, 200), sun_screen, 18, 2)
            # Little rays around sun
            for deg in range(0, 360, 45):
                ang = math.radians(deg)
                ex  = int(sun_screen[0] + 24 * math.cos(ang))
                ey  = int(sun_screen[1] + 24 * math.sin(ang))
                pygame.draw.line(surf, SUN_COLOR, sun_screen, (ex, ey), 1)

        # Incident ray
        if on_screen(sun_screen, dp[0]) and on_screen(centre_screen, dp[1]):
            pygame.draw.line(surf, RAY_COLOR, sun_screen, centre_screen, 1)

        # ── Reflected Ray ─────────────────────────────────────────────────
        # Use the normal of the face closest to the sun direction
        # We approximate with the overall object normal facing the sun
        D     = centre - sun
        D_hat = D / (np.linalg.norm(D) + 1e-10)

        # Pick the face whose normal best aligns with -D_hat (faces sun)
        best_n   = None
        best_dot = -np.inf
        for face in self.faces:
            if len(face) < 3:
                continue
            n = face_normal(world_verts[face[0]],
                            world_verts[face[1]],
                            world_verts[face[2]])
            # LA: DOT PRODUCT to test alignment
            d = np.dot(-D_hat, n)
            if d > best_dot:
                best_dot = d
                best_n   = n

        if best_n is not None:
            # LA: VECTOR REFLECTION  R = D − 2(D·n̂)n̂
            R      = reflect_ray(D_hat, best_n)
            R_end  = centre + R * 2.5    # extend reflected ray into scene

            r_pts  = np.array([centre, R_end], dtype=np.float64)
            r_sp, r_dp = project_to_screen(r_pts, self.VP_mat, SCENE_WIDTH, HEIGHT)
            r0 = tuple(r_sp[0].astype(int))
            r1 = tuple(r_sp[1].astype(int))

            if on_screen(r0, r_dp[0]) and on_screen(r1, r_dp[1]):
                pygame.draw.line(surf, REFLECT_COLOR, r0, r1, 2)
                # Arrowhead
                dx, dy = r1[0] - r0[0], r1[1] - r0[1]
                ang    = math.atan2(dy, dx)
                for da in [0.4, -0.4]:
                    ax = int(r1[0] - 10 * math.cos(ang + da))
                    ay = int(r1[1] - 10 * math.sin(ang + da))
                    pygame.draw.line(surf, REFLECT_COLOR, r1, (ax, ay), 2)

    def _draw_legend(self, surf: pygame.Surface):
        """On-screen legend explaining the visual elements."""
        items = [
            (RAY_COLOR,     "Incident Ray (from Sun)"),
            (REFLECT_COLOR, "Reflected Ray  R=D−2(D·n̂)n̂"),
            (SUN_COLOR,     "Sun (Light Source)"),
            ((80,80,80),    "Shadow (Plane Projection)"),
        ]
        y = HEIGHT - 100
        for color, text in items:
            pygame.draw.rect(surf, color, (12, y, 14, 10))
            t = self.font_sm.render(text, True, TEXT_COLOR)
            surf.blit(t, (32, y - 1))
            y += 20

    def _draw_ui_panel(self, surf: pygame.Surface):
        """Render the right-hand control panel."""
        panel_rect = pygame.Rect(SCENE_WIDTH, 0, UI_PANEL_WIDTH, HEIGHT)
        pygame.draw.rect(surf, PANEL_COLOR, panel_rect)
        pygame.draw.line(surf, PANEL_BORDER, (SCENE_WIDTH, 0), (SCENE_WIDTH, HEIGHT), 2)

        # Title
        title = self.font_lg.render("⚙  CONTROLS", True, ACCENT_COLOR)
        surf.blit(title, (SCENE_WIDTH + 14, 14))

        # Section headers — positions come from the layout cursor in __init__
        def section(label, y):
            t = self.font_md.render(label, True, (100, 130, 200))
            surf.blit(t, (SCENE_WIDTH + 14, y))

        section("── Object ──────────────", self._sec_object_y)
        section("── Sun (Light) ──────────", self._sec_sun_y)
        section("── Shapes ───────────────", self._sec_shapes_y)

        for sl in self.sliders.values():
            sl.draw(surf, self.font_sm)

        self.btn_cube.draw(surf, self.font_sm)
        self.btn_pyramid.draw(surf, self.font_sm)
        self.btn_prism.draw(surf, self.font_sm)
        self.btn_rotate.draw(surf, self.font_sm)

        # FPS
        fps_t = self.font_sm.render(f"FPS: {int(self.clock.get_fps())}", True, (80,100,140))
        surf.blit(fps_t, (SCENE_WIDTH + 14, HEIGHT - 24))

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    self.auto_rotate = not self.auto_rotate

            for sl in self.sliders.values():
                sl.handle_event(event)

            if self.btn_cube.handle_event(event):
                self.verts, self.faces = make_cube()
                self.shape_idx = 0
            if self.btn_pyramid.handle_event(event):
                self.verts, self.faces = make_pyramid()
                self.shape_idx = 1
            if self.btn_prism.handle_event(event):
                self.verts, self.faces = make_prism()
                self.shape_idx = 2
            if self.btn_rotate.handle_event(event):
                self.auto_rotate = not self.auto_rotate
                self.btn_rotate.label = (
                    "Auto-Rot ON" if self.auto_rotate else "Auto-Rot OFF")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            self.handle_events()

            if self.auto_rotate:
                self.rot_y += dt * 0.6     # gentle continuous rotation

            # ── Render ────────────────────────────────────────────────────
            scene_surf = pygame.Surface((SCENE_WIDTH, HEIGHT))
            scene_surf.fill(BG_COLOR)

            # Compute world-space vertices for this frame
            # (builds and applies the model matrix — MATRIX MULTIPLICATION)
            world_verts = self._world_vertices()

            # 1. Floor / Ground plane grid
            self._draw_floor(scene_surf)

            # 2. Shadow (ground-plane projection of object vertices)
            self._draw_shadow(world_verts, scene_surf)

            # 3. Object (face shading with diffuse lighting via DOT PRODUCT)
            self._draw_object(world_verts, scene_surf)

            # 4. Sun + incident ray + reflected ray
            self._draw_sun_and_rays(world_verts, scene_surf)

            # 5. Legend
            self._draw_legend(scene_surf)

            # Shape name overlay
            name_t = self.font_md.render(
                f"Shape: {self.shape_names[self.shape_idx]}  |  "
                f"R to toggle rotation  |  ESC to quit",
                True, (60, 80, 120))
            scene_surf.blit(name_t, (14, 10))

            # ── Compose frame ─────────────────────────────────────────────
            self.screen.blit(scene_surf, (0, 0))
            self._draw_ui_panel(self.screen)

            pygame.display.flip()


# =============================================================================
# ░░  ENTRY POINT  ░░
# =============================================================================

if __name__ == "__main__":
    Engine3D().run()
