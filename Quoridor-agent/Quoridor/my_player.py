"""Quoridor agent: iterative-deepening principal-variation search on a bitboard.

Cells are bits of one integer (bit ``r * 9 + c``). Two masks hold the open
edges: ``vmask`` bit ``i`` means ``i <-> i + 9`` is open, ``hmask`` bit ``i``
means ``i <-> i + 1`` is open. Wall anchors also live on the 9-stride grid
(``WALL_GRID``, where column 8 acts as a row-wrap guard bit), allowing exact
shortest-path bridge/pair wall cuts and geometric wall legality to be computed
in pure bitwise operations without per-edge set or list allocations.
"""

from __future__ import annotations

import json
import os
import random
import time

from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor


INF = 1_000_000_000
MATE = 100_000
MATE_BOUND = 90_000
FORCED_RACE = 40_000

EXACT = 0
LOWER = 1
UPPER = 2

# Move codes: 0..80 pawn destination, 100..163 horizontal wall, 200..263 vertical wall.
H_BASE = 100
V_BASE = 200
MOVE_LIMIT = 264

STEP = 120
# Half-step tempo aligns odd and even plies (STEP - TEMPO == TEMPO) so LMR,
# depth extensions, aspiration windows, and qsearch stand-pat never oscillate.
TEMPO = STEP // 2

# ---- Wall pricing -------------------------------------------------------
# Worth of the walls still in hand, indexed by count. Every wall is priced
# above a single pawn step (120) so early walls are not dumped lightly, while
# late walls stay below 2.1x a step to avoid hoarding in losing endgames.
# Marginals, tenth wall to first: 135 145 155 165 175 185 200 215 230 245.
WALL_VALUE = (0, 245, 475, 690, 890, 1075, 1250, 1415, 1570, 1715, 1850)
# A root pawn move that lengthens our own path must beat the others by this much.
REGRESS_PENALTY = 45

# ---- Corridor & trap pricing --------------------------------------------
THREAT_CAP = 8
THREAT_STRIKE = 45
THREAT_STRIKE_HIGH = 30
THREAT_DODGE = 25
THREAT_DODGE_HIGH = 20
DODGE_LOCKED_EXTRA = 18
MAX_STRIKE_BONUS = 4 * THREAT_STRIKE + (THREAT_CAP - 5) * THREAT_STRIKE_HIGH
MAX_DODGE_PENALTY = 4 * THREAT_DODGE + (THREAT_CAP - 5) * THREAT_DODGE_HIGH + DODGE_LOCKED_EXTRA

# ---- Repetition ---------------------------------------------------------
# The harness has no draw rule: a shuffle ends only when a clock runs out.
CONTEMPT = 80
CONTEMPT_BEHIND_CLOCK = 3 * STEP
CONTEMPT_PER_VISIT = STEP
CONTEMPT_CAP = 20_000
LOOP_BAN_VISITS = 4
REPEAT_BUDGET = 0.2

# ---- Time ---------------------------------------------------------------
TIME_PLAN_SHARE = 0.40
TIME_PLAN_PLY = 40
CAP_SHARP = 28 / 900
CAP_SHARP_FEW_WALLS = 18 / 900
CAP_ORDINARY = 14 / 900
CAP_OPENING = 6 / 900
CAP_OVER_PLAN = 10 / 900
CONTACT_RESERVE = 140 / 900

TT_LIMIT = 1_200_000
DAG_CACHE_LIMIT = 500_000
DIST_CACHE_LIMIT = 800_000
THREAT_CACHE_LIMIT = 250_000

MOVE_LOG = os.environ.get("QUORIDOR_MOVE_LOG", "0") == "1"

FULL_V = (1 << 72) - 1
FULL_H = sum(1 << (r * 9 + c) for r in range(9) for c in range(8))
WALL_GRID = FULL_H & FULL_V
GOAL_MASK = tuple(sum(1 << (r * 9 + c) for c in range(9)) for r in range(9))
ROW = tuple(i // 9 for i in range(81))
COL = tuple(i % 9 for i in range(81))
CENTER_DIST = tuple(abs((i % 9) - 4) for i in range(81))

# Edge bits each wall closes: horizontal walls clear ``vmask``, vertical ones ``hmask``.
H_CLR = tuple(3 << ((i >> 3) * 9 + (i & 7)) for i in range(64))
V_CLR = tuple((1 << ((i >> 3) * 9 + (i & 7))) | (1 << ((i >> 3) * 9 + (i & 7) + 9)) for i in range(64))

try:
    _popcount = int.bit_count
except AttributeError:  # Python < 3.10
    def _popcount(x: int) -> int:
        return bin(x).count("1")


def _zobrist_tables():
    rng = random.Random(0xC0FFEE)

    def rnd():
        return rng.getrandbits(64)

    pawns = [[rnd() for _ in range(81)] for _ in range(2)]
    wall_z = [0] * MOVE_LIMIT
    for idx in range(64):
        wall_z[H_BASE + idx] = rnd()
        wall_z[V_BASE + idx] = rnd()
    walls_left = [[rnd() for _ in range(11)] for _ in range(2)]
    side = rnd()
    goal = rnd()
    return pawns, tuple(wall_z), walls_left, side, goal


Z_PAWN, Z_WALL, Z_LEFT, Z_SIDE, Z_GOAL = _zobrist_tables()


def _wall_tables():
    """Precompute stride-9 bitboard geometry, mask-keepers, touch masks, and conflicts."""
    w_row = [0] * MOVE_LIMIT
    w_col = [0] * MOVE_LIMIT
    w_anchor = [0] * MOVE_LIMIT
    w_gbit = [0] * MOVE_LIMIT
    w_wkey = [0] * MOVE_LIMIT
    w_v_keep = [FULL_V] * MOVE_LIMIT
    w_h_keep = [FULL_H] * MOVE_LIMIT
    w_on_edge = [False] * MOVE_LIMIT
    w_touch_hg = [0] * MOVE_LIMIT
    w_touch_vg = [0] * MOVE_LIMIT
    w_conflicts: list[tuple[int, ...]] = [() for _ in range(MOVE_LIMIT)]
    anchor_to_h = [0] * 72
    anchor_to_v = [0] * 72

    for idx in range(64):
        r, c = divmod(idx, 8)
        a = r * 9 + c
        gbit = 1 << a

        # Horizontal wall: grid segment y = r + 1, x in [c, c + 2].
        hw = H_BASE + idx
        anchor_to_h[a] = hw
        w_row[hw] = r
        w_col[hw] = c
        w_anchor[hw] = a
        w_gbit[hw] = gbit
        w_wkey[hw] = gbit << 72
        w_v_keep[hw] = FULL_V & ~H_CLR[idx]
        w_on_edge[hw] = c == 0 or c == 7
        thg = 0
        if c >= 2:
            thg |= 1 << (r * 9 + (c - 2))
        if c <= 5:
            thg |= 1 << (r * 9 + (c + 2))
        tvg = 0
        for vr in range(max(0, r - 1), min(7, r + 1) + 1):
            for vc in range(max(0, c - 1), min(7, c + 1) + 1):
                if vr == r and vc == c:
                    continue
                tvg |= 1 << (vr * 9 + vc)
        w_touch_hg[hw] = thg
        w_touch_vg[hw] = tvg
        conf = [V_BASE + idx]
        if c > 0:
            conf.append(H_BASE + idx - 1)
        if c < 7:
            conf.append(H_BASE + idx + 1)
        w_conflicts[hw] = tuple(conf)

        # Vertical wall: grid segment x = c + 1, y in [r, r + 2].
        vw = V_BASE + idx
        anchor_to_v[a] = vw
        w_row[vw] = r
        w_col[vw] = c
        w_anchor[vw] = a
        w_gbit[vw] = gbit
        w_wkey[vw] = gbit
        w_h_keep[vw] = FULL_H & ~V_CLR[idx]
        w_on_edge[vw] = r == 0 or r == 7
        tvg2 = 0
        if r >= 2:
            tvg2 |= 1 << ((r - 2) * 9 + c)
        if r <= 5:
            tvg2 |= 1 << ((r + 2) * 9 + c)
        thg2 = 0
        for wr in range(max(0, r - 1), min(7, r + 1) + 1):
            for wc in range(max(0, c - 1), min(7, c + 1) + 1):
                if wr == r and wc == c:
                    continue
                thg2 |= 1 << (wr * 9 + wc)
        w_touch_hg[vw] = thg2
        w_touch_vg[vw] = tvg2
        conf2 = [H_BASE + idx]
        if r > 0:
            conf2.append(V_BASE + idx - 8)
        if r < 7:
            conf2.append(V_BASE + idx + 8)
        w_conflicts[vw] = tuple(conf2)

    return (
        tuple(w_row),
        tuple(w_col),
        tuple(w_anchor),
        tuple(w_gbit),
        tuple(w_wkey),
        tuple(w_v_keep),
        tuple(w_h_keep),
        tuple(w_on_edge),
        tuple(w_touch_hg),
        tuple(w_touch_vg),
        tuple(w_conflicts),
        tuple(anchor_to_h),
        tuple(anchor_to_v),
    )


(
    W_ROW,
    W_COL,
    W_ANCHOR,
    W_GBIT,
    W_WKEY,
    W_V_KEEP,
    W_H_KEEP,
    W_ON_EDGE,
    W_TOUCH_HG,
    W_TOUCH_VG,
    W_CONFLICTS,
    ANCHOR_TO_H,
    ANCHOR_TO_V,
) = _wall_tables()


def _wall_worth(walls: int, enemy_d: int) -> int:
    """Wall reserve value with a symmetric endgame usability taper when enemy_d <= 2."""
    if enemy_d <= 2 and walls > 2:
        usable = WALL_VALUE[2]
        return usable + ((WALL_VALUE[walls] - usable) >> 2)
    return WALL_VALUE[walls]


class SearchTimeout(Exception):
    """Raised when the move deadline is reached."""


class Engine:
    """Mutable bitboard used inside the search. ``load`` rebuilds it each turn."""

    __slots__ = (
        "vmask", "hmask", "h_grid", "v_grid", "wkey", "pawns", "goals",
        "walls_left", "side", "ply", "zhash", "rcache", "dag_cache", "dcache",
        "tcache", "tt", "history", "countermoves", "killers", "deadline",
        "nodes", "ext", "root_pawn", "completed_depth", "completed_score",
        "line", "played", "contempt", "banned",
    )

    def __init__(self, dag_cache: dict, tt: dict, history: list[int], countermoves: list[int]) -> None:
        self.rcache: dict[int, tuple[int | None, int, int]] = {}
        self.dag_cache = dag_cache
        self.dcache: dict[int, int] = {}
        self.tcache: dict[int, tuple[tuple[int, int], ...]] = {}
        self.tt = tt
        self.history = history
        self.countermoves = countermoves
        self.vmask = FULL_V
        self.hmask = FULL_H
        self.h_grid = 0
        self.v_grid = 0
        self.wkey = 0
        self.pawns = [0, 0]
        self.goals = [0, 8]
        self.walls_left = [10, 10]
        self.side = 0
        self.ply = 0
        self.zhash = 0
        self.killers = [[-1, -1] for _ in range(96)]
        self.deadline = 0.0
        self.nodes = 0
        self.ext = 0
        self.root_pawn = 0
        self.completed_depth = 0
        self.completed_score = 0
        self.line: set[int] = set()
        self.played: dict[int, int] = {}
        self.contempt = CONTEMPT
        self.banned: set[int] = set()

    # ------------------------------------------------------------------
    # Board
    # ------------------------------------------------------------------

    def load(self, state: GameStateQuoridor, me: PlayerQuoridor) -> None:
        opponent = state._opponent(me)
        self.goals = [me.get_goal_row(), opponent.get_goal_row()]
        mine = state.rep.pawn_positions[me.id]
        theirs = state.rep.pawn_positions[opponent.id]
        self.pawns = [mine[0] * 9 + mine[1], theirs[0] * 9 + theirs[1]]
        self.root_pawn = self.pawns[0]
        left = state.rep.remaining_walls
        self.walls_left = [left[me.id], left[opponent.id]]
        self.h_grid = 0
        self.v_grid = 0
        self.vmask = FULL_V
        self.hmask = FULL_H
        self.side = 0
        self.ply = 0
        self.ext = 0

        z = Z_PAWN[0][self.pawns[0]] ^ Z_PAWN[1][self.pawns[1]]
        z ^= Z_LEFT[0][self.walls_left[0]] ^ Z_LEFT[1][self.walls_left[1]]
        if self.goals[0] == 8:
            z ^= Z_GOAL

        for wall in state.rep.walls:
            if wall.orientation == Orientation.HORIZONTAL:
                if 1 <= wall.row <= 8 and 0 <= wall.col <= 7:
                    r = wall.row - 1
                    c = wall.col
                    move = H_BASE + r * 8 + c
                    self.h_grid |= W_GBIT[move]
                    self.vmask &= W_V_KEEP[move]
                    z ^= Z_WALL[move]
            elif 0 <= wall.row <= 7 and 1 <= wall.col <= 8:
                r = wall.row
                c = wall.col - 1
                move = V_BASE + r * 8 + c
                self.v_grid |= W_GBIT[move]
                self.hmask &= W_H_KEEP[move]
                z ^= Z_WALL[move]

        self.wkey = (self.h_grid << 72) | self.v_grid
        self.zhash = z

    def _forbid_masks(self) -> tuple[int, int]:
        """71-bit stride-9 masks of geometrically illegal H and V wall anchors."""
        hg = self.h_grid
        vg = self.v_grid
        forbid_h = (hg | (hg << 1) | (hg >> 1) | vg) & WALL_GRID
        forbid_v = (vg | (vg << 9) | (vg >> 9) | hg) & WALL_GRID
        return forbid_h, forbid_v

    def geometry_ok(self, move: int) -> bool:
        gbit = W_GBIT[move]
        hg = self.h_grid
        vg = self.v_grid
        if move < V_BASE:
            return (((hg | (hg << 1) | (hg >> 1) | vg) & gbit) == 0)
        return (((vg | (vg << 9) | (vg >> 9) | hg) & gbit) == 0)

    def apply(self, move: int):
        snap = (
            self.pawns[0], self.pawns[1], self.h_grid, self.v_grid, self.wkey,
            self.vmask, self.hmask, self.walls_left[0], self.walls_left[1],
            self.side, self.zhash, self.ply,
        )
        side = self.side
        won = False
        if move < 81:
            self.zhash ^= Z_PAWN[side][self.pawns[side]] ^ Z_PAWN[side][move]
            self.pawns[side] = move
            won = ROW[move] == self.goals[side]
        elif move < V_BASE:
            left = self.walls_left[side]
            self.zhash ^= Z_WALL[move] ^ Z_LEFT[side][left] ^ Z_LEFT[side][left - 1]
            self.walls_left[side] = left - 1
            self.h_grid |= W_GBIT[move]
            self.wkey |= W_WKEY[move]
            self.vmask &= W_V_KEEP[move]
        else:
            left = self.walls_left[side]
            self.zhash ^= Z_WALL[move] ^ Z_LEFT[side][left] ^ Z_LEFT[side][left - 1]
            self.walls_left[side] = left - 1
            self.v_grid |= W_GBIT[move]
            self.wkey |= W_WKEY[move]
            self.hmask &= W_H_KEEP[move]
        self.zhash ^= Z_SIDE
        self.side ^= 1
        self.ply += 1
        return snap, won

    def undo(self, token) -> None:
        (
            self.pawns[0], self.pawns[1], self.h_grid, self.v_grid, self.wkey,
            self.vmask, self.hmask, self.walls_left[0], self.walls_left[1],
            self.side, self.zhash, self.ply,
        ) = token[0]

    # ------------------------------------------------------------------
    # Paths & Bitboard DAG
    # ------------------------------------------------------------------

    def _dist_masks(self, start: int, goal_row: int, v: int, h: int, wkey: int) -> int | None:
        """Pawn-ignoring distance on the given edge masks, ``None`` if unreachable."""
        key = (wkey << 11) | (goal_row << 7) | start
        dcache = self.dcache
        hit = dcache.get(key, -2)
        if hit != -2:
            return None if hit < 0 else hit
        goal = GOAL_MASK[goal_row]
        reached = 1 << start
        d = 0
        while not (reached & goal):
            grown = (
                reached
                | ((reached >> 9) & v)
                | ((reached & v) << 9)
                | ((reached >> 1) & h)
                | ((reached & h) << 1)
            )
            if grown == reached:
                if len(dcache) < DIST_CACHE_LIMIT:
                    dcache[key] = -1
                return None
            reached = grown
            d += 1
        if len(dcache) < DIST_CACHE_LIMIT:
            dcache[key] = d
        return d

    def _dist(self, start: int, goal_row: int) -> int | None:
        return self._dist_masks(start, goal_row, self.vmask, self.hmask, self.wkey)

    def _race_masks(self, start: int, goal_row: int, v: int, h: int, wkey: int) -> tuple[int | None, int, int]:
        """Fast leaf shortest-path ``(d, narrow, step_mask)`` without wall-edge scans."""
        key = (wkey << 11) | (goal_row << 7) | start
        rcache = self.rcache
        hit = rcache.get(key)
        if hit is not None:
            return hit
        goal = GOAL_MASK[goal_row]
        front = 1 << start
        if front & goal:
            res = (0, 9, 0)
            if len(rcache) < DIST_CACHE_LIMIT:
                rcache[key] = res
            return res
        layers = [front]
        seen = front
        res = (None, 0, 0)
        while True:
            front = (
                ((front >> 9) & v)
                | ((front & v) << 9)
                | ((front >> 1) & h)
                | ((front & h) << 1)
            ) & ~seen
            if not front:
                break
            back = front & goal
            if back:
                narrow = _popcount(back)
                k = len(layers) - 1
                while k > 0:
                    back = layers[k] & (
                        ((back >> 9) & v)
                        | ((back & v) << 9)
                        | ((back >> 1) & h)
                        | ((back & h) << 1)
                    )
                    width = _popcount(back)
                    if width < narrow:
                        narrow = width
                    k -= 1
                res = (len(layers), narrow, back)
                break
            layers.append(front)
            seen |= front
        if len(rcache) < DIST_CACHE_LIMIT:
            rcache[key] = res
        return res

    def _race(self, start: int, goal_row: int) -> tuple[int | None, int, int]:
        return self._race_masks(start, goal_row, self.vmask, self.hmask, self.wkey)

    def _dag_masks(self, start: int, goal_row: int, v: int, h: int, wkey: int):
        """Unified shortest-path DAG analysis on ``(v, h)``:

        Returns ``(d, narrow, step_mask, h_all, v_all, h_cut, v_cut)`` where:
        - ``d``: shortest-path length (``None`` if unreachable)
        - ``narrow``: minimum vertex width across interior/goal DAG layers
        - ``step_mask``: 81-bit mask of all ``d - 1`` adjacent first steps from ``start``
        - ``h_all, v_all``: 71-bit stride-9 masks of all walls touching any DAG edge
        - ``h_cut, v_cut``: exact-complete 71-bit stride-9 masks of walls capable of
          cutting ALL shortest paths (``delta_d >= 1`` or disconnecting). Any wall
          outside ``(h_cut, v_cut)`` is mathematically guaranteed to leave ``d`` unchanged.
        """
        key = (wkey << 11) | (goal_row << 7) | start
        dag_cache = self.dag_cache
        hit = dag_cache.get(key)
        if hit is not None:
            return hit

        goal = GOAL_MASK[goal_row]
        front = 1 << start
        if front & goal:
            res = (0, 9, 0, 0, 0, 0, 0)
            if len(dag_cache) < DAG_CACHE_LIMIT:
                dag_cache[key] = res
            return res

        layers = [front]
        seen = front
        while True:
            front = (
                ((front >> 9) & v)
                | ((front & v) << 9)
                | ((front >> 1) & h)
                | ((front & h) << 1)
            ) & ~seen
            if not front:
                res = (None, 0, 0, 0, 0, 0, 0)
                if len(dag_cache) < DAG_CACHE_LIMIT:
                    dag_cache[key] = res
                return res
            if front & goal:
                break
            layers.append(front)
            seen |= front

        upper = front & goal
        narrow = _popcount(upper)
        all_v = 0
        all_h = 0
        h_cut = 0
        v_cut = 0
        same_hv = 0
        same_vv = 0
        step_mask = 0

        for k in range(len(layers) - 1, -1, -1):
            if k > 0:
                lower = layers[k] & (
                    ((upper >> 9) & v)
                    | ((upper & v) << 9)
                    | ((upper >> 1) & h)
                    | ((upper & h) << 1)
                )
                width = _popcount(lower)
                if width < narrow:
                    narrow = width
            else:
                step_mask = upper
                lower = layers[0]

            ve = v & ((lower & (upper >> 9)) | (upper & (lower >> 9)))
            he = h & ((lower & (upper >> 1)) | (upper & (lower >> 1)))
            all_v |= ve
            all_h |= he

            if he == 0:
                if ve and (ve & (ve - 1)) == 0:
                    h_cut |= ve | (ve >> 1)
                else:
                    pair = ve & (ve >> 1)
                    if pair and (ve & ~(pair | (pair << 1))) == 0:
                        h_cut |= pair
                    else:
                        same_hv |= pair
            elif ve == 0:
                if (he & (he - 1)) == 0:
                    v_cut |= he | (he >> 9)
                else:
                    pair = he & (he >> 9)
                    if pair and (he & ~(pair | (pair << 9))) == 0:
                        v_cut |= pair
                    else:
                        same_vv |= pair
            else:
                same_hv |= ve & (ve >> 1)
                same_vv |= he & (he >> 9)

            upper = lower

        h_all = (all_v | (all_v >> 1)) & WALL_GRID
        v_all = (all_h | (all_h >> 9)) & WALL_GRID
        h_cut = (h_cut | ((all_v & (all_v >> 1)) & ~same_hv)) & WALL_GRID
        v_cut = (v_cut | ((all_h & (all_h >> 9)) & ~same_vv)) & WALL_GRID
        d = len(layers)
        res = (d, narrow, step_mask, h_all, v_all, h_cut, v_cut)
        if len(dag_cache) < DAG_CACHE_LIMIT:
            dag_cache[key] = res
        if len(self.rcache) < DIST_CACHE_LIMIT:
            self.rcache[key] = (d, narrow, step_mask)
        if len(self.dcache) < DIST_CACHE_LIMIT:
            self.dcache[key] = d
        return res

    def _dag(self, start: int, goal_row: int):
        return self._dag_masks(start, goal_row, self.vmask, self.hmask, self.wkey)

    def _best_first_step(self, step_mask: int, goal_row: int) -> int:
        best_step = -1
        best_key = -1000
        bits = step_mask
        while bits:
            lsb = bits & -bits
            bits ^= lsb
            cell = lsb.bit_length() - 1
            key = -abs(ROW[cell] - goal_row) * 10 - CENTER_DIST[cell]
            if key > best_key:
                best_key = key
                best_step = cell
        return best_step

    def _raw_threats(self, start: int, goal_row: int, dag_info=None) -> tuple[tuple[int, int], ...]:
        """Sorted ``(delta_d, wall)`` for geometrically legal walls that lengthen ``start``'s path."""
        key = (self.wkey << 11) | (goal_row << 7) | start
        tcache = self.tcache
        hit = tcache.get(key)
        if hit is not None:
            return hit
        if dag_info is None:
            dag_info = self._dag(start, goal_row)
        d = dag_info[0]
        if not d:
            return ()

        forbid_h, forbid_v = self._forbid_masks()
        hc = dag_info[5] & ~forbid_h
        vc = dag_info[6] & ~forbid_v
        if not hc and not vc:
            if len(tcache) < THREAT_CACHE_LIMIT:
                tcache[key] = ()
            return ()

        raw: list[tuple[int, int]] = []
        v, h = self.vmask, self.hmask
        goal = GOAL_MASK[goal_row]
        start_bit = 1 << start

        bits = hc
        while bits:
            lsb = bits & -bits
            bits ^= lsb
            wall = ANCHOR_TO_H[lsb.bit_length() - 1]
            nv = v & W_V_KEEP[wall]
            front = seen = start_bit
            nd = 0
            while front:
                front = (
                    ((front >> 9) & nv)
                    | ((front & nv) << 9)
                    | ((front >> 1) & h)
                    | ((front & h) << 1)
                ) & ~seen
                nd += 1
                if front & goal:
                    if nd > d:
                        raw.append((nd - d, wall))
                    break
                seen |= front

        bits = vc
        while bits:
            lsb = bits & -bits
            bits ^= lsb
            wall = ANCHOR_TO_V[lsb.bit_length() - 1]
            nh = h & W_H_KEEP[wall]
            front = seen = start_bit
            nd = 0
            while front:
                front = (
                    ((front >> 9) & v)
                    | ((front & v) << 9)
                    | ((front >> 1) & nh)
                    | ((front & nh) << 1)
                ) & ~seen
                nd += 1
                if front & goal:
                    if nd > d:
                        raw.append((nd - d, wall))
                    break
                seen |= front

        if len(raw) > 1:
            raw.sort(reverse=True)
        res = tuple(raw)
        if len(tcache) < THREAT_CACHE_LIMIT:
            tcache[key] = res
        return res

    def _threat_info(
        self,
        defender: int,
        defender_goal: int,
        attacker: int,
        attacker_goal: int,
        att_d: int,
    ) -> tuple[int, int]:
        """Return ``(max_net_detour, wall_mdist_to_defender)`` for a single legal wall."""
        raw = self._raw_threats(defender, defender_goal)
        if not raw:
            return 0, 99

        att_dag = None
        v, h = self.vmask, self.hmask
        att_goal_mask = GOAL_MASK[attacker_goal]
        att_bit = 1 << attacker
        def_r, def_c = ROW[defender], COL[defender]

        worst = 0
        best_mdist = 99
        for delta_def, wall in raw:
            if delta_def < worst:
                break
            if att_dag is None:
                att_dag = self._dag(attacker, attacker_goal)
            gbit = W_GBIT[wall]
            cuts_att = bool((att_dag[5] if wall < V_BASE else att_dag[6]) & gbit)
            if cuts_att:
                nv = v & W_V_KEEP[wall]
                nh = h & W_H_KEEP[wall]
                front = seen = att_bit
                na = 0
                ok = False
                while front:
                    front = (
                        ((front >> 9) & nv)
                        | ((front & nv) << 9)
                        | ((front >> 1) & nh)
                        | ((front & nh) << 1)
                    ) & ~seen
                    na += 1
                    if front & att_goal_mask:
                        ok = True
                        break
                    seen |= front
                if not ok:
                    continue
                net = delta_def - (na - att_d)
            else:
                net = delta_def

            if net > worst or (net == worst and net > 0):
                wr, wc = W_ROW[wall], W_COL[wall]
                dr = 0 if wr <= def_r <= wr + 1 else (wr - def_r if def_r < wr else def_r - (wr + 1))
                dc = 0 if wc <= def_c <= wc + 1 else (wc - def_c if def_c < wc else def_c - (wc + 1))
                md = dr + dc
                if net > worst or md > best_mdist:
                    worst = net
                    best_mdist = md
        return worst, best_mdist

    def _cell_dist(self, start: int, target: int) -> int | None:
        """Pawn-ignoring steps between two cells."""
        if start == target:
            return 0
        v, h = self.vmask, self.hmask
        key = (self.wkey << 15) | (1 << 14) | (start << 7) | target
        dcache = self.dcache
        hit = dcache.get(key, -2)
        if hit != -2:
            return None if hit < 0 else hit
        goal = 1 << target
        front = seen = 1 << start
        d = 0
        found = -1
        while True:
            front = (
                ((front >> 9) & v)
                | ((front & v) << 9)
                | ((front >> 1) & h)
                | ((front & h) << 1)
            ) & ~seen
            if not front:
                break
            d += 1
            if front & goal:
                found = d
                break
            seen |= front
        if len(dcache) < DIST_CACHE_LIMIT:
            dcache[key] = found
        return None if found < 0 else found

    def _pawns_interact(self, d0: int, d1: int) -> bool:
        """True unless the pawns mathematically cannot meet before the leader finishes."""
        leader = d0 if d0 <= d1 else d1
        sep = self._cell_dist(self.pawns[0], self.pawns[1])
        if sep is None or sep > 2 * leader:
            return False
        return True

    def _pawns_facing(self) -> bool:
        """True until the pawns have passed: one still stands between the other and its goal."""
        r0 = ROW[self.pawns[0]]
        r1 = ROW[self.pawns[1]]
        return (r1 - r0) * (self.goals[0] - r0) > 0 or (r0 - r1) * (self.goals[1] - r1) > 0

    def distances(self) -> tuple[int | None, int | None]:
        return (
            self._race(self.pawns[0], self.goals[0])[0],
            self._race(self.pawns[1], self.goals[1])[0],
        )

    def _open(self, a: int, b: int) -> bool:
        d = b - a
        if d == 9:
            return ((self.vmask >> a) & 1) == 1
        if d == -9:
            return ((self.vmask >> b) & 1) == 1
        if d == 1:
            return ((self.hmask >> a) & 1) == 1
        return ((self.hmask >> b) & 1) == 1

    def _open_adj_mask(self, cell: int) -> int:
        """Bitmask of open orthogonal neighbors of ``cell`` (ignoring pawns)."""
        v = self.vmask
        h = self.hmask
        mask = 0
        if cell >= 9 and ((v >> (cell - 9)) & 1):
            mask |= 1 << (cell - 9)
        if (v >> cell) & 1:
            mask |= 1 << (cell + 9)
        if cell > 0 and ((h >> (cell - 1)) & 1):
            mask |= 1 << (cell - 1)
        if (h >> cell) & 1:
            mask |= 1 << (cell + 1)
        return mask

    def _pawn_destinations(self) -> list[int]:
        side = self.side
        start = self.pawns[side]
        opp = self.pawns[side ^ 1]
        v = self.vmask
        h = self.hmask

        # Fast path (>92% of nodes): pawns are not orthogonally adjacent.
        diff = start - opp
        if diff != 9 and diff != -9 and diff != 1 and diff != -1:
            res: list[int] = []
            if start >= 9 and ((v >> (start - 9)) & 1):
                res.append(start - 9)
            if (v >> start) & 1:
                res.append(start + 9)
            if start > 0 and ((h >> (start - 1)) & 1):
                res.append(start - 1)
            if (h >> start) & 1:
                res.append(start + 1)
            return res

        r, c = ROW[start], COL[start]
        found: list[int] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            found.extend(self._try_dir(r, c, dr, dc))
        if len(found) < 2:
            return found
        unique: list[int] = []
        for dest in found:
            if dest not in unique:
                unique.append(dest)
        return unique

    def _try_dir(self, r: int, c: int, dr: int, dc: int) -> list[int]:
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= 9 or nc < 0 or nc >= 9:
            return []
        a = r * 9 + c
        b = nr * 9 + nc
        if not self._open(a, b):
            return []
        opp = self.pawns[self.side ^ 1]
        if b == opp:
            jr = nr + dr
            jc = nc + dc
            if 0 <= jr < 9 and 0 <= jc < 9 and self._open(b, jr * 9 + jc):
                return self._try_dir(nr, nc, dr, dc)
            if dr == 0:
                return self._try_dir(nr, nc, -1, 0) + self._try_dir(nr, nc, 1, 0)
            return self._try_dir(nr, nc, 0, -1) + self._try_dir(nr, nc, 0, 1)
        return [b]

    def _can_win_in_one(self) -> bool:
        side = self.side
        p = self.pawns[side]
        goal = self.goals[side]
        if abs(ROW[p] - goal) > 2:
            return False
        for dest in self._pawn_destinations():
            if ROW[dest] == goal:
                return True
        return False

    def _winning_destination(self) -> int | None:
        side = self.side
        p = self.pawns[side]
        goal = self.goals[side]
        if abs(ROW[p] - goal) > 2:
            return None
        for dest in self._pawn_destinations():
            if ROW[dest] == goal:
                return dest
        return None

    # ------------------------------------------------------------------
    # Evaluation & Move Generation
    # ------------------------------------------------------------------

    def evaluate(self, alpha: int = -INF, beta: int = INF) -> int:
        if self._can_win_in_one():
            return MATE - (self.ply + 1)

        side = self.side
        opp = side ^ 1
        r0 = self._race(self.pawns[0], self.goals[0])
        r1 = self._race(self.pawns[1], self.goals[1])
        d0, d1 = r0[0], r1[0]
        if side == 0:
            (my_d, my_narrow, my_steps), (opp_d, opp_narrow, opp_steps) = r0, r1
        else:
            (my_d, my_narrow, my_steps), (opp_d, opp_narrow, opp_steps) = r1, r0

        if my_d is None:
            return -MATE + self.ply
        if opp_d is None:
            return MATE - self.ply

        my_left = self.walls_left[side]
        opp_left = self.walls_left[opp]
        adv = opp_d - my_d

        if opp_left == 0 and adv >= 0 and not self._pawns_interact(d0, d1):
            return FORCED_RACE + 10_000 - my_d * 100 + opp_d
        if my_left == 0 and adv < 0 and not self._pawns_interact(d0, d1):
            return -(FORCED_RACE + 10_000) + opp_d * 100 - my_d

        my_p = self.pawns[side]
        opp_p = self.pawns[opp]
        score = (
            adv * STEP
            + TEMPO
            + _wall_worth(my_left, opp_d)
            - _wall_worth(opp_left, my_d)
            + (CENTER_DIST[opp_p] - CENTER_DIST[my_p]) * 4
        )

        # Exact pawn-interaction physics when orthogonally adjacent with an open edge:
        # unified bonus for forward jumps (+60) and penalty for dead-end body-blocks (-55/-110).
        p_mdist = abs(ROW[my_p] - ROW[opp_p]) + abs(COL[my_p] - COL[opp_p])
        if p_mdist == 1 and self._open(my_p, opp_p):
            my_goal = self.goals[side]
            best_next_d = 99
            for dest in self._pawn_destinations():
                if (my_steps >> dest) & 1:
                    if my_d - 1 < best_next_d:
                        best_next_d = my_d - 1
                else:
                    nd = self._dist(dest, my_goal)
                    if nd is not None and nd < best_next_d:
                        best_next_d = nd
            if best_next_d <= my_d - 2:
                score += STEP // 2
            elif best_next_d == my_d:
                score -= 55
            elif best_next_d < 99 and best_next_d > my_d:
                score -= 110

        # Sound lazy bounds before evaluating single-wall corridor threats.
        check_dodge = bool(opp_left and my_narrow <= 2)
        check_strike = bool(my_left and opp_narrow <= 2)
        max_neg = MAX_DODGE_PENALTY if check_dodge else 0
        max_pos = MAX_STRIKE_BONUS if check_strike else 0
        if score - max_neg >= beta:
            return score - max_neg
        if score + max_pos <= alpha:
            return score + max_pos

        if check_dodge:
            threat, w_mdist = self._threat_info(
                my_p, self.goals[side],
                opp_p, self.goals[opp], opp_d,
            )
            if threat >= 2:
                capped = min(threat, THREAT_CAP)
                base_pen = (min(capped, 5) - 1) * THREAT_DODGE + max(0, capped - 5) * THREAT_DODGE_HIGH
                if w_mdist >= 2 and capped >= 3:
                    base_pen += DODGE_LOCKED_EXTRA
                score -= base_pen
            elif my_narrow <= 1:
                score -= 16
            if score >= beta:
                return score
            if score + max_pos <= alpha:
                return score + max_pos

        if check_strike:
            threat, _w_mdist = self._threat_info(
                opp_p, self.goals[opp],
                my_p, self.goals[side], my_d,
            )
            if threat >= 2:
                capped = min(threat, THREAT_CAP)
                score += (min(capped, 5) - 1) * THREAT_STRIKE + max(0, capped - 5) * THREAT_STRIKE_HIGH
            elif opp_narrow <= 1:
                score += 16

        return score

    def _collect_walls(self, depth: int, prev_move: int = -1) -> tuple[list[int], list[int], bool]:
        """Generate ordered ``(strong_walls, setup_walls)`` using bitboard DAG cuts."""
        side = self.side
        my_left = self.walls_left[side]
        if my_left <= 0:
            return [], [], False
        opp = side ^ 1
        pawns = self.pawns
        goals = self.goals
        my_dag = self._dag(pawns[side], goals[side])
        opp_dag = self._dag(pawns[opp], goals[opp])
        my_d, my_narrow = my_dag[0], my_dag[1]
        opp_d, opp_narrow = opp_dag[0], opp_dag[1]
        if my_d is None or opp_d is None:
            return [], [], False

        if self.walls_left[opp] == 0 and opp_d >= my_d:
            d0, d1 = (my_d, opp_d) if side == 0 else (opp_d, my_d)
            if not self._pawns_interact(d0, d1):
                return [], [], False

        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

        forbid_h, forbid_v = self._forbid_masks()
        opp_h_all = opp_dag[3] & ~forbid_h
        opp_v_all = opp_dag[4] & ~forbid_v
        opp_h_cut = opp_dag[5] & ~forbid_h
        opp_v_cut = opp_dag[6] & ~forbid_v
        my_h_cut = my_dag[5]
        my_v_cut = my_dag[6]

        root = self.ply == 0
        critical = opp_d <= 2
        # Offensive 0-gain setup walls require >= 2 walls in hand so a follow-up exists.
        allow_off_setup = (root or depth >= 2) and my_left >= 2
        allow_def_setup = root or depth >= 2
        allow_zero_gain = (root or depth >= 2) and (opp_d <= 5 or opp_narrow <= 2)

        vmask, hmask, wkey = self.vmask, self.hmask, self.wkey
        h_grid, v_grid = self.h_grid, self.v_grid
        dist_masks = self._dist_masks
        opp_r, opp_c = ROW[pawns[opp]], COL[pawns[opp]]
        cm_move = self.countermoves[prev_move] if 0 <= prev_move < MOVE_LIMIT else -1

        tactical: list[tuple[int, int]] = []
        setup: list[tuple[int, int]] = []
        synergy_seeds: list[tuple[int, int, int, int]] = []
        has_pos_gain = False

        # 1) Walls that cut ALL current shortest paths of opponent (only ~2..6 walls!).
        for cut_bits, to_move, my_cut_mask in (
            (opp_h_cut, ANCHOR_TO_H, my_h_cut),
            (opp_v_cut, ANCHOR_TO_V, my_v_cut),
        ):
            bits = cut_bits
            while bits:
                lsb = bits & -bits
                bits ^= lsb
                wall = to_move[lsb.bit_length() - 1]
                nv = vmask & W_V_KEEP[wall]
                nh = hmask & W_H_KEEP[wall]
                nwkey = wkey | W_WKEY[wall]
                new_opp = dist_masks(pawns[opp], goals[opp], nv, nh, nwkey)
                if new_opp is None:
                    continue
                if my_cut_mask & lsb:
                    new_my = dist_masks(pawns[side], goals[side], nv, nh, nwkey)
                    if new_my is None:
                        continue
                else:
                    new_my = my_d

                delta_opp = new_opp - opp_d
                delta_my = new_my - my_d
                gain = delta_opp - delta_my
                hist_bonus = (min(self.history[wall], 4_000) >> 4) + (6_000 if wall == cm_move else 0)

                wr, wc = W_ROW[wall], W_COL[wall]
                dr = 0 if wr <= opp_r <= wr + 1 else (wr - opp_r if opp_r < wr else opp_r - (wr + 1))
                dc = 0 if wc <= opp_c <= wc + 1 else (wc - opp_c if opp_c < wc else opp_c - (wc + 1))
                mdist = dr + dc
                touches = W_ON_EDGE[wall] or bool(
                    (h_grid & W_TOUCH_HG[wall]) | (v_grid & W_TOUCH_VG[wall])
                )

                if delta_opp <= 2 and mdist <= 3:
                    synergy_seeds.append((-mdist * 10 + delta_opp * 5, wall, delta_opp, gain))

                if critical and delta_opp >= 1:
                    has_pos_gain = True
                    prio = 1_800_000 + delta_opp * 10_000 - delta_my * 1_000 - mdist * 500 + hist_bonus
                    tactical.append((prio, wall))
                elif delta_opp >= 1 and gain >= 1:
                    # Demote isolated far-away +1 horizon walls so close/connected walls
                    # are never pushed out of tac_limit by floating goal-row blockers.
                    if gain == 1 and delta_opp == 1 and not touches and mdist >= 3 and opp_d >= 4:
                        prio = 1_150_000 - mdist * 4_000 + hist_bonus
                    else:
                        has_pos_gain = True
                        prio = (
                            1_500_000
                            + gain * 12_000
                            + delta_opp * 500
                            - delta_my * 200
                            + (3_500 if touches else 0)
                            - mdist * 1_500
                            + hist_bonus
                        )
                    tactical.append((prio, wall))
                elif delta_opp >= 1 and gain == 0 and allow_zero_gain:
                    # Close connected enclosure walls (e.g. delta_opp >= 2 right on opp's nose)
                    # rank ahead of floating far-away +1 noise.
                    close_seal = 240_000 if (delta_opp >= 2 and mdist <= 1 and touches) else 0
                    prio = (
                        1_120_000
                        + close_seal
                        + delta_opp * 6_000
                        + (3_000 if touches else 0)
                        - mdist * 1_500
                        + hist_bonus
                    )
                    tactical.append((prio, wall))
                elif allow_off_setup and delta_opp == 0 and delta_my == 0:
                    if touches or mdist <= 2:
                        prio = 750_000 + hist_bonus + (8_000 if touches else 0) + max(0, 4 - mdist) * 2_000
                        setup.append((prio, wall))

        # 2) Non-cutting DAG walls: guaranteed delta_opp == 0 in O(1) without BFS!
        if allow_off_setup:
            for noncut_bits, to_move, my_cut_mask in (
                (opp_h_all & ~opp_h_cut, ANCHOR_TO_H, my_h_cut),
                (opp_v_all & ~opp_v_cut, ANCHOR_TO_V, my_v_cut),
            ):
                bits = noncut_bits
                while bits:
                    lsb = bits & -bits
                    bits ^= lsb
                    wall = to_move[lsb.bit_length() - 1]
                    wr, wc = W_ROW[wall], W_COL[wall]
                    dr = 0 if wr <= opp_r <= wr + 1 else (wr - opp_r if opp_r < wr else opp_r - (wr + 1))
                    dc = 0 if wc <= opp_c <= wc + 1 else (wc - opp_c if opp_c < wc else opp_c - (wc + 1))
                    mdist = dr + dc
                    touches = W_ON_EDGE[wall] or bool(
                        (h_grid & W_TOUCH_HG[wall]) | (v_grid & W_TOUCH_VG[wall])
                    )
                    if mdist > 3 and not (touches and opp_narrow <= 3):
                        continue
                    # Skip if it hurts or disconnects our own path.
                    if my_cut_mask & lsb:
                        new_my = dist_masks(
                            pawns[side],
                            goals[side],
                            vmask & W_V_KEEP[wall],
                            hmask & W_H_KEEP[wall],
                            wkey | W_WKEY[wall],
                        )
                        if new_my != my_d:
                            continue

                    narrows = False
                    new_narrow = opp_narrow
                    if (root or depth >= 3) and opp_narrow >= 2 and (mdist <= 2 or (touches and mdist <= 3)):
                        new_narrow = self._race_masks(
                            pawns[opp],
                            goals[opp],
                            vmask & W_V_KEEP[wall],
                            hmask & W_H_KEEP[wall],
                            wkey | W_WKEY[wall],
                        )[1]
                        narrows = new_narrow < opp_narrow

                    if narrows or (touches and mdist <= 3) or mdist <= 2:
                        hist_bonus = (min(self.history[wall], 4_000) >> 4) + (6_000 if wall == cm_move else 0)
                        prio = 750_000 + hist_bonus
                        if narrows:
                            prio += 20_000 * (opp_narrow - new_narrow)
                            if new_narrow <= 1:
                                prio += 15_000
                        if touches:
                            prio += 8_000
                        if mdist <= 3:
                            prio += (4 - mdist) * 2_000
                        setup.append((prio, wall))

        # 3) 2-Wall Backdoor / Escape-Sealing Synergy Traps (near root: ply <= 2).
        # Finds d+1 / d+2 escape walls outside ``opp_all`` whose combination with a
        # main-path blocker creates a super-additive box/siding detour.
        if self.ply <= 2 and (root or depth >= 4) and my_left >= 2 and 2 <= opp_d <= 8 and synergy_seeds:
            synergy_seeds.sort(reverse=True)
            syn_found: dict[int, int] = {}
            for _seed_score, w_main, d_main, g_main in synergy_seeds[:2]:
                v1 = vmask & W_V_KEEP[w_main]
                h1 = hmask & W_H_KEEP[w_main]
                wkey1 = wkey | W_WKEY[w_main]
                hg1 = h_grid | (W_GBIT[w_main] if w_main < V_BASE else 0)
                vg1 = v_grid | (W_GBIT[w_main] if w_main >= V_BASE else 0)
                fh1 = (hg1 | (hg1 << 1) | (hg1 >> 1) | vg1) & WALL_GRID
                fv1 = (vg1 | (vg1 << 9) | (vg1 >> 9) | hg1) & WALL_GRID
                dag1 = self._dag_masks(pawns[opp], goals[opp], v1, h1, wkey1)
                if dag1[0] is None:
                    continue
                for back_bits, to_move, my_cut_mask in (
                    (dag1[5] & ~fh1, ANCHOR_TO_H, my_h_cut),
                    (dag1[6] & ~fv1, ANCHOR_TO_V, my_v_cut),
                ):
                    bits = back_bits
                    while bits:
                        lsb = bits & -bits
                        bits ^= lsb
                        w_back = to_move[lsb.bit_length() - 1]
                        # Require enclosure geometry: touches an existing wall, w_main, or board edge.
                        if not (
                            W_ON_EDGE[w_back]
                            or (hg1 & W_TOUCH_HG[w_back])
                            or (vg1 & W_TOUCH_VG[w_back])
                        ):
                            continue
                        d_both = dist_masks(
                            pawns[opp],
                            goals[opp],
                            v1 & W_V_KEEP[w_back],
                            h1 & W_H_KEEP[w_back],
                            wkey1 | W_WKEY[w_back],
                        )
                        if d_both is None:
                            continue
                        total_opp_gain = d_both - opp_d
                        if total_opp_gain - d_main < 3:
                            continue
                        # Verify w_back alone is legal on the current board and check our path.
                        vb = vmask & W_V_KEEP[w_back]
                        hb = hmask & W_H_KEEP[w_back]
                        wkeyb = wkey | W_WKEY[w_back]
                        if (opp_h_cut if w_back < V_BASE else opp_v_cut) & lsb:
                            if dist_masks(pawns[opp], goals[opp], vb, hb, wkeyb) is None:
                                continue
                        if my_cut_mask & lsb:
                            my_after_b = dist_masks(pawns[side], goals[side], vb, hb, wkeyb)
                            if my_after_b is None or my_after_b - my_d > 1:
                                continue
                        else:
                            my_after_b = my_d
                        net_combo = total_opp_gain - (my_after_b - my_d)
                        if net_combo >= max(3, g_main + 2):
                            prio = 1_320_000 + net_combo * 8_000 + (min(self.history[w_back], 4_000) >> 4)
                            if prio > syn_found.get(w_back, 0):
                                syn_found[w_back] = prio
            if syn_found:
                top_syn = sorted(((p, w) for w, p in syn_found.items()), reverse=True)[:2]
                tactical.extend(top_syn)

        # 4) Defensive / anti-trap conflict walls when our own path is narrow.
        defensive: list[tuple[int, int]] = []
        if self.walls_left[opp] > 0 and my_narrow <= 2 and allow_def_setup:
            raw_my = self._raw_threats(pawns[side], goals[side], my_dag)
            if raw_my:
                threats: list[tuple[int, int]] = []
                for d_my, tw in raw_my:
                    if d_my < 2:
                        break
                    gbit_tw = W_GBIT[tw]
                    if (opp_h_cut if tw < V_BASE else opp_v_cut) & gbit_tw:
                        new_opp_t = dist_masks(
                            pawns[opp],
                            goals[opp],
                            vmask & W_V_KEEP[tw],
                            hmask & W_H_KEEP[tw],
                            wkey | W_WKEY[tw],
                        )
                        if new_opp_t is None:
                            continue
                        net_t = d_my - (new_opp_t - opp_d)
                    else:
                        net_t = d_my
                    if net_t >= 2:
                        threats.append((net_t, tw))
                if threats:
                    threats.sort(reverse=True)
                    def_seen: set[int] = set()
                    for net_t, tw in threats[:3]:
                        for wall in W_CONFLICTS[tw]:
                            if wall in def_seen or not self.geometry_ok(wall):
                                continue
                            def_seen.add(wall)
                            gbit = W_GBIT[wall]
                            nv = vmask & W_V_KEEP[wall]
                            nh = hmask & W_H_KEEP[wall]
                            nwkey = wkey | W_WKEY[wall]
                            if (my_h_cut if wall < V_BASE else my_v_cut) & gbit:
                                new_my = dist_masks(pawns[side], goals[side], nv, nh, nwkey)
                                if new_my != my_d:
                                    continue
                            if (opp_h_cut if wall < V_BASE else opp_v_cut) & gbit:
                                new_opp = dist_masks(pawns[opp], goals[opp], nv, nh, nwkey)
                                if new_opp is None:
                                    continue
                                delta_opp = new_opp - opp_d
                            else:
                                delta_opp = 0
                            prio = (
                                1_250_000
                                + net_t * 10_000
                                + delta_opp * 5_000
                                + (min(self.history[wall], 4_000) >> 4)
                            )
                            defensive.append((prio, wall))

        if not tactical and not defensive and not setup:
            return [], [], False

        tactical.sort(reverse=True)
        defensive.sort(reverse=True)
        setup.sort(reverse=True)

        if root:
            tac_limit, def_limit, setup_limit = 14, 3, 4
        elif critical:
            tac_limit, def_limit, setup_limit = 10, 2, 2
        elif depth >= 8:
            tac_limit, def_limit, setup_limit = 7, 3, 3
        elif depth >= 5:
            tac_limit, def_limit, setup_limit = 5, 2, 2
        elif depth >= 3:
            tac_limit, def_limit, setup_limit = 4, 2, 2
        else:
            tac_limit, def_limit, setup_limit = 4, 2, 1

        strong_scored: list[tuple[int, int]] = []
        used: set[int] = set()
        for bucket, lim in ((tactical, tac_limit), (defensive, def_limit)):
            count = 0
            for item in bucket:
                w = item[1]
                if w in used:
                    continue
                used.add(w)
                strong_scored.append(item)
                count += 1
                if count >= lim:
                    break
        strong_scored.sort(reverse=True)
        strong_walls = [w for _p, w in strong_scored]

        setup_walls: list[int] = []
        count = 0
        for _p, w in setup:
            if w in used:
                continue
            used.add(w)
            setup_walls.append(w)
            count += 1
            if count >= setup_limit:
                break

        return strong_walls, setup_walls, has_pos_gain

    def _split_pawn_moves(self, my_dag, skip_move: int = -1, prev_move: int = -1) -> tuple[list[int], list[int]]:
        """Split legal pawn moves into ``(advancing_moves, non_advancing_moves)``."""
        side = self.side
        goal = self.goals[side]
        my_d = my_dag[0]
        step_mask = my_dag[2]
        start = self.pawns[side]
        cm_move = self.countermoves[prev_move] if 0 <= prev_move < MOVE_LIMIT else -1

        adv: list[tuple[int, int]] = []
        quiet: list[tuple[int, int]] = []
        for dest in self._pawn_destinations():
            if dest == skip_move:
                continue
            bonus = (min(self.history[dest], 4_000) >> 4) + (5_000 if dest == cm_move else 0)
            if ROW[dest] == goal:
                adv.append((3_000_000 + bonus, dest))
            elif (step_mask >> dest) & 1:
                adv.append((2_000_000 - CENTER_DIST[dest] * 8 + bonus, dest))
            elif abs(ROW[dest] - ROW[start]) + abs(COL[dest] - COL[start]) > 1:
                # Jump destination: check exact post-jump distance.
                d_dest = self._dist(dest, goal)
                if my_d is not None and d_dest is not None and d_dest < my_d:
                    adv.append((2_100_000 + (my_d - d_dest) * 100_000 - CENTER_DIST[dest] * 8 + bonus, dest))
                else:
                    quiet.append((950_000 - abs(ROW[dest] - goal) * 1_000 - CENTER_DIST[dest] * 8 + bonus, dest))
            else:
                quiet.append((900_000 - abs(ROW[dest] - goal) * 1_000 - CENTER_DIST[dest] * 8 + bonus, dest))

        if len(adv) > 1:
            adv.sort(reverse=True)
        if len(quiet) > 1:
            quiet.sort(reverse=True)
        return [m for _p, m in adv], [m for _p, m in quiet]

    def generate(self, depth: int, tt_move: int) -> list[int]:
        """Full ordered candidate list for the root search."""
        side = self.side
        my_race = self._race(self.pawns[side], self.goals[side])
        adv_pawns, quiet_pawns = self._split_pawn_moves(my_race)
        strong_walls, setup_walls, _has_pos = self._collect_walls(depth)

        ordered = adv_pawns + strong_walls + quiet_pawns + setup_walls

        if tt_move >= 0 and self._plausible(tt_move):
            if tt_move in ordered:
                ordered.remove(tt_move)
            ordered.insert(0, tt_move)

        ply = self.ply if self.ply < len(self.killers) else len(self.killers) - 1
        insert_at = 1 if ordered and ordered[0] == tt_move else 0
        for killer in self.killers[ply]:
            if killer < 0 or killer == tt_move:
                continue
            if killer < 81:
                if not self._plausible(killer):
                    continue
                if killer in ordered:
                    ordered.remove(killer)
                ordered.insert(insert_at, killer)
                insert_at += 1
            elif killer in ordered:
                ordered.remove(killer)
                ordered.insert(insert_at, killer)
                insert_at += 1
        return ordered

    def _plausible(self, move: int) -> bool:
        if move < 81:
            return move in self._pawn_destinations()
        if self.walls_left[self.side] <= 0 or not self.geometry_ok(move):
            return False
        gbit = W_GBIT[move]
        dag0 = self._dag(self.pawns[0], self.goals[0])
        dag1 = self._dag(self.pawns[1], self.goals[1])
        nv = self.vmask & W_V_KEEP[move]
        nh = self.hmask & W_H_KEEP[move]
        nwkey = self.wkey | W_WKEY[move]
        if (dag0[5] if move < V_BASE else dag0[6]) & gbit:
            if self._dist_masks(self.pawns[0], self.goals[0], nv, nh, nwkey) is None:
                return False
        if (dag1[5] if move < V_BASE else dag1[6]) & gbit:
            if self._dist_masks(self.pawns[1], self.goals[1], nv, nh, nwkey) is None:
                return False
        return True

    # ------------------------------------------------------------------
    # Tactical Quiescence & Principal-Variation Search
    # ------------------------------------------------------------------

    def qsearch(self, alpha: int, beta: int, qdepth: int) -> int:
        """Bounded tactical quiescence: resolves ``opp_d == 1`` goal-threats and pawn jumps."""
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

        if self._can_win_in_one():
            return MATE - (self.ply + 1)

        zhash = self.zhash
        if zhash in self.line or zhash in self.played:
            return -self.contempt if self.side == 0 else self.contempt

        entry = self.tt.get(zhash)
        if entry is not None:
            _tt_depth, tt_score, tt_flag, _tt_move = entry
            if tt_score > MATE_BOUND:
                tt_score -= self.ply
            elif tt_score < -MATE_BOUND:
                tt_score += self.ply
            if tt_flag == EXACT or (tt_flag == LOWER and tt_score >= beta) or (
                tt_flag == UPPER and tt_score <= alpha
            ):
                return tt_score

        side = self.side
        opp = side ^ 1
        opp_p = self.pawns[opp]
        opp_goal = self.goals[opp]

        # Mandatory check-like evasion when opponent is 1 step from the goal row:
        # never stand pat on an unblockable 1-turn mate threat.
        if abs(ROW[opp_p] - opp_goal) <= 1 and qdepth >= -1:
            opp_dag = self._dag(opp_p, opp_goal)
            if opp_dag[0] == 1:
                my_dag = self._dag(self.pawns[side], self.goals[side])
                if my_dag[0] is None:
                    return -MATE + self.ply
                evasions: list[int] = []
                if self.walls_left[side] > 0:
                    fh, fv = self._forbid_masks()
                    my_hc, my_vc = my_dag[5], my_dag[6]
                    for cbits, to_move, mcut in (
                        (opp_dag[5] & ~fh, ANCHOR_TO_H, my_hc),
                        (opp_dag[6] & ~fv, ANCHOR_TO_V, my_vc),
                    ):
                        bits = cbits
                        while bits:
                            lsb = bits & -bits
                            bits ^= lsb
                            w = to_move[lsb.bit_length() - 1]
                            nv = self.vmask & W_V_KEEP[w]
                            nh = self.hmask & W_H_KEEP[w]
                            nwkey = self.wkey | W_WKEY[w]
                            if self._dist_masks(opp_p, opp_goal, nv, nh, nwkey) is None:
                                continue
                            if (mcut & lsb) and self._dist_masks(
                                self.pawns[side], self.goals[side], nv, nh, nwkey
                            ) is None:
                                continue
                            evasions.append(w)
                evasions.extend(self._pawn_destinations())
                if not evasions:
                    return -MATE + self.ply
                best = -INF
                self.line.add(zhash)
                try:
                    for move in evasions:
                        token = self.apply(move)
                        try:
                            if token[1]:
                                score = MATE - self.ply
                            else:
                                score = -self.qsearch(-beta, -alpha, qdepth - 1)
                        finally:
                            self.undo(token)
                        if score > best:
                            best = score
                        if best > alpha:
                            alpha = best
                        if alpha >= beta:
                            break
                finally:
                    self.line.discard(zhash)
                return best

        stand = self.evaluate(alpha, beta)
        if stand >= beta or qdepth <= 0:
            return stand
        if stand > alpha:
            alpha = stand

        # Resolve immediate forward pawn jumps (pawns orthogonally adjacent with open edge).
        my_p = self.pawns[side]
        if abs(ROW[my_p] - ROW[opp_p]) + abs(COL[my_p] - COL[opp_p]) == 1 and self._open(my_p, opp_p):
            my_goal = self.goals[side]
            my_d = self._race(my_p, my_goal)[0]
            if my_d is not None:
                jumps: list[int] = []
                for dest in self._pawn_destinations():
                    if abs(ROW[dest] - ROW[my_p]) + abs(COL[dest] - COL[my_p]) > 1:
                        nd = self._dist(dest, my_goal)
                        if nd is not None and nd <= my_d - 2:
                            jumps.append(dest)
                if jumps:
                    self.line.add(zhash)
                    try:
                        for move in jumps:
                            token = self.apply(move)
                            try:
                                if token[1]:
                                    score = MATE - self.ply
                                else:
                                    score = -self.qsearch(-beta, -alpha, qdepth - 1)
                            finally:
                                self.undo(token)
                            if score > stand:
                                stand = score
                            if stand > alpha:
                                alpha = stand
                            if alpha >= beta:
                                break
                    finally:
                        self.line.discard(zhash)

        return stand

    def negamax(self, depth: int, alpha: int, beta: int, prev_move: int = -1) -> int:
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

        if self._can_win_in_one():
            return MATE - (self.ply + 1)

        zhash = self.zhash
        if zhash in self.line or zhash in self.played:
            return -self.contempt if self.side == 0 else self.contempt

        extended = False
        opp = self.side ^ 1
        opp_d: int | None = None
        if 0 < depth < 6 and self.ext < 2 and self.walls_left[self.side] > 0:
            if abs(ROW[self.pawns[opp]] - self.goals[opp]) <= 3:
                opp_d = self._race(self.pawns[opp], self.goals[opp])[0]
                if opp_d is not None and opp_d <= 2:
                    depth += 1
                    self.ext += 1
                    extended = True

        alpha_orig = alpha
        tt_move = -1
        entry = self.tt.get(zhash)
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_score > MATE_BOUND:
                tt_score -= self.ply
            elif tt_score < -MATE_BOUND:
                tt_score += self.ply
            if tt_depth >= depth:
                if tt_flag == EXACT or (tt_flag == LOWER and tt_score >= beta) or (
                    tt_flag == UPPER and tt_score <= alpha
                ):
                    if extended:
                        self.ext -= 1
                    return tt_score

        if depth <= 0:
            side = self.side
            my_p = self.pawns[side]
            opp_p = self.pawns[opp]
            opp_goal = self.goals[opp]
            # Only invoke tactical qsearch on 1-step goal threats or adjacent pawn jump contact.
            if abs(ROW[opp_p] - opp_goal) <= 1 or (
                abs(ROW[my_p] - ROW[opp_p]) + abs(COL[my_p] - COL[opp_p]) == 1
                and self._open(my_p, opp_p)
            ):
                return self.qsearch(alpha, beta, 2)
            return self.evaluate(alpha, beta)

        self.line.add(zhash)
        try:
            best = -INF
            best_move = -1
            tried_tt = -1
            node_ply = self.ply
            move_idx = 0
            had_line_rep = False

            # Stage 1: Hash move first. A beta cutoff skips all move generation.
            if tt_move >= 0 and self._plausible(tt_move):
                tried_tt = tt_move
                token = self.apply(tt_move)
                try:
                    if token[1]:
                        score = MATE - self.ply
                    else:
                        if self.zhash in self.line:
                            had_line_rep = True
                        score = -self.negamax(depth - 1, -beta, -alpha, tt_move)
                finally:
                    self.undo(token)
                best = score
                best_move = tt_move
                move_idx = 1
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self._record_cutoff(node_ply, tt_move, prev_move, depth)

            # Stage 2: Pawn moves before wall generation (skips _collect_walls on cutoffs).
            # Advancing pawn moves get full depth; aimless non-advancing steps are LMR-reduced
            # so they do not bloat the tree or starve Stage-3 walls.
            if alpha < beta:
                side = self.side
                my_race = self._race(self.pawns[side], self.goals[side])
                if my_race[0] is None:
                    return -MATE + self.ply
                adv_pawns, quiet_pawns = self._split_pawn_moves(my_race, tried_tt, prev_move)

                urgent_quiet = (
                    not adv_pawns
                    or (self.walls_left[opp] > 0 and my_race[1] <= 2)
                    or (
                        abs(ROW[self.pawns[0]] - ROW[self.pawns[1]])
                        + abs(COL[self.pawns[0]] - COL[self.pawns[1]])
                        <= 2
                    )
                )
                cm_move = self.countermoves[prev_move] if 0 <= prev_move < MOVE_LIMIT else -1

                for move in adv_pawns:
                    first = move_idx == 0
                    token = self.apply(move)
                    try:
                        if token[1]:
                            score = MATE - self.ply
                        else:
                            if self.zhash in self.line:
                                had_line_rep = True
                            if first:
                                score = -self.negamax(depth - 1, -beta, -alpha, move)
                            else:
                                score = -self.negamax(depth - 1, -alpha - 1, -alpha, move)
                                if alpha < score < beta:
                                    score = -self.negamax(depth - 1, -beta, -alpha, move)
                    finally:
                        self.undo(token)
                    move_idx += 1
                    if score > best:
                        best = score
                        best_move = move
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        self._record_cutoff(node_ply, move, prev_move, depth)
                        break

                if alpha < beta and quiet_pawns:
                    for move in quiet_pawns:
                        first = move_idx == 0
                        reduce_pawn = depth >= 3 and move_idx >= 1 and not urgent_quiet and move != cm_move
                        token = self.apply(move)
                        try:
                            if token[1]:
                                score = MATE - self.ply
                            else:
                                if self.zhash in self.line:
                                    had_line_rep = True
                                if first:
                                    score = -self.negamax(depth - 1, -beta, -alpha, move)
                                elif reduce_pawn:
                                    score = -self.negamax(depth - 2, -alpha - 1, -alpha, move)
                                    if score > alpha:
                                        score = -self.negamax(depth - 1, -alpha - 1, -alpha, move)
                                        if alpha < score < beta:
                                            score = -self.negamax(depth - 1, -beta, -alpha, move)
                                else:
                                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, move)
                                    if alpha < score < beta:
                                        score = -self.negamax(depth - 1, -beta, -alpha, move)
                        finally:
                            self.undo(token)
                        move_idx += 1
                        if score > best:
                            best = score
                            best_move = move
                        if best > alpha:
                            alpha = best
                        if alpha >= beta:
                            self._record_cutoff(node_ply, move, prev_move, depth)
                            break

            # Stage 3: Wall moves if pawn moves did not cause a beta cutoff.
            # Dedicated wall_idx guarantees the #1 strongest wall (and primary killer) gets full
            # depth - 1 instead of being LMR-poisoned by prior pawn moves, while keeping the
            # remaining walls tightly reduced.
            if alpha < beta and self.walls_left[self.side] > 0:
                strong_walls, setup_walls, has_pos_gain = self._collect_walls(depth, prev_move)
                walls = strong_walls + setup_walls
                if walls:
                    kply = node_ply if node_ply < len(self.killers) else len(self.killers) - 1
                    killers_here = self.killers[kply]
                    cm_move = self.countermoves[prev_move] if 0 <= prev_move < MOVE_LIMIT else -1
                    insert_at = 0
                    for promo in (killers_here[0], killers_here[1], cm_move):
                        if promo >= 81 and promo != tried_tt and promo in walls:
                            walls.remove(promo)
                            walls.insert(insert_at, promo)
                            insert_at += 1

                    if opp_d is None:
                        opp_d = self._race(self.pawns[opp], self.goals[opp])[0]
                    lmr_ok = depth >= 3 and (opp_d is None or opp_d > 2)
                    n_strong = len(strong_walls)
                    wall_idx = 0
                    for move in walls:
                        if move == tried_tt:
                            continue
                        first = move_idx == 0
                        unreduced_wall = (wall_idx == 0 and has_pos_gain) or move == killers_here[0]
                        token = self.apply(move)
                        try:
                            if self.zhash in self.line:
                                had_line_rep = True
                            if first:
                                score = -self.negamax(depth - 1, -beta, -alpha, move)
                            elif lmr_ok and not unreduced_wall:
                                r_depth = (
                                    depth - 3
                                    if (depth >= 6 and wall_idx >= n_strong + 1)
                                    else depth - 2
                                )
                                score = -self.negamax(r_depth, -alpha - 1, -alpha, move)
                                if score > alpha:
                                    score = -self.negamax(depth - 1, -alpha - 1, -alpha, move)
                                    if alpha < score < beta:
                                        score = -self.negamax(depth - 1, -beta, -alpha, move)
                            else:
                                score = -self.negamax(depth - 1, -alpha - 1, -alpha, move)
                                if alpha < score < beta:
                                    score = -self.negamax(depth - 1, -beta, -alpha, move)
                        finally:
                            self.undo(token)
                        move_idx += 1
                        wall_idx += 1
                        if score > best:
                            best = score
                            best_move = move
                        if best > alpha:
                            alpha = best
                        if alpha >= beta:
                            self._record_cutoff(node_ply, move, prev_move, depth)
                            break

            if best_move < 0:
                return self.evaluate()

            if best <= alpha_orig:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT

            # Do not store path-dependent line-repetition scores (unless a clean beta cutoff occurred).
            if (
                best != self.contempt
                and best != -self.contempt
                and (not had_line_rep or flag == LOWER)
            ):
                stored = best
                if stored > MATE_BOUND:
                    stored += self.ply
                elif stored < -MATE_BOUND:
                    stored -= self.ply
                previous = self.tt.get(zhash)
                if previous is None or depth >= previous[0]:
                    # Preserve a prior EXACT/LOWER best_move when an ALL-node fails low (UPPER).
                    store_move = (
                        previous[3]
                        if (flag == UPPER and previous is not None and previous[3] >= 0)
                        else best_move
                    )
                    if len(self.tt) > TT_LIMIT:
                        self._age_tt()
                    self.tt[zhash] = (depth, stored, flag, store_move)
            return best
        finally:
            self.line.discard(zhash)
            if extended:
                self.ext -= 1

    def _record_cutoff(self, ply: int, move: int, prev_move: int, depth: int) -> None:
        self.history[move] += depth * depth
        if 0 <= prev_move < MOVE_LIMIT:
            self.countermoves[prev_move] = move
        if ply < len(self.killers) and move >= 81:
            first = self.killers[ply][0]
            if move != first:
                self.killers[ply][1] = first
                self.killers[ply][0] = move

    def _age_tt(self) -> None:
        """Drop shallow entries first; deep entries hold the expensive work."""
        tt = self.tt
        for keep in (3, 5, 8):
            for key in [key for key, entry in tt.items() if entry[0] < keep]:
                del tt[key]
            if len(tt) <= TT_LIMIT // 2:
                return
        tt.clear()

    def _root_bias(self, move: int, my_d: int | None, opp_d: int | None = None) -> int:
        """Discourage regressive pawn shuffles and isolated far back-rank +1 wall dumps."""
        if move < 81:
            if my_d is None:
                return 0
            after = self._dist(move, self.goals[0])
            if after is not None and after > my_d:
                return -REGRESS_PENALTY
            return 0
        if opp_d is not None and opp_d >= 4:
            touches = W_ON_EDGE[move] or bool(
                (self.h_grid & W_TOUCH_HG[move]) | (self.v_grid & W_TOUCH_VG[move])
            )
            if not touches:
                opp_r, opp_c = ROW[self.pawns[1]], COL[self.pawns[1]]
                wr, wc = W_ROW[move], W_COL[move]
                dr = 0 if wr <= opp_r <= wr + 1 else (wr - opp_r if opp_r < wr else opp_r - (wr + 1))
                dc = 0 if wc <= opp_c <= wc + 1 else (wc - opp_c if opp_c < wc else opp_c - (wc + 1))
                mdist = dr + dc
                if mdist >= 3:
                    new_opp = self._dist_masks(
                        self.pawns[1],
                        self.goals[1],
                        self.vmask & W_V_KEEP[move],
                        self.hmask & W_H_KEEP[move],
                        self.wkey | W_WKEY[move],
                    )
                    if new_opp is not None and new_opp - opp_d <= 1:
                        return -min(45, 15 * (mdist - 1))
        return 0

    def _child_hash(self, move: int) -> int:
        token = self.apply(move)
        child = self.zhash
        self.undo(token)
        return child

    def _is_banned(self, move: int) -> bool:
        return bool(self.banned) and self._child_hash(move) in self.banned

    def root_moves(self, depth: int, tt_move: int) -> list[int]:
        """Root candidates without moves that close a loop, unless nothing else is left."""
        moves = self.generate(depth, tt_move)
        if self.banned:
            kept = [move for move in moves if not self._is_banned(move)]
            if kept:
                return kept
        return moves

    def search_root(
        self,
        depth: int,
        alpha: int = -INF,
        beta: int = INF,
        opening_variety: bool = False,
        rescue_out: list | None = None,
    ):
        entry = self.tt.get(self.zhash)
        tt_move = entry[3] if entry is not None else -1
        moves = self.root_moves(depth, tt_move)
        if not moves:
            raise RuntimeError("No legal move.")
        my_d, opp_d = self.distances()

        alpha_orig = alpha
        narrowed = alpha_orig > -INF or beta < INF
        best_move = None
        best_score = -INF
        move0_score = None
        move_scores: list[tuple[int, int, bool]] = []

        for index, move in enumerate(moves):
            if time.perf_counter() >= self.deadline and best_move is not None:
                if depth >= 2 or narrowed:
                    raise SearchTimeout
                break
            bias = self._root_bias(move, my_d, opp_d)
            raw_alpha = alpha - bias if -MATE_BOUND < alpha < MATE_BOUND else alpha
            raw_beta = beta - bias if -MATE_BOUND < beta < MATE_BOUND else beta
            token = self.apply(move)
            try:
                if token[1]:
                    score = MATE - self.ply
                    is_exact = True
                elif index == 0:
                    score = -self.negamax(depth - 1, -raw_beta, -raw_alpha, move)
                    is_exact = raw_alpha < score < raw_beta
                elif opening_variety and -MATE_BOUND < raw_alpha < MATE_BOUND:
                    band = raw_alpha - 16
                    score = -self.negamax(depth - 1, -band - 1, -band, move)
                    if score > band:
                        score = -self.negamax(depth - 1, -raw_beta, -band, move)
                        is_exact = band < score < raw_beta
                    else:
                        is_exact = False
                else:
                    score = -self.negamax(depth - 1, -raw_alpha - 1, -raw_alpha, move)
                    if raw_alpha < score < raw_beta:
                        score = -self.negamax(depth - 1, -raw_beta, -raw_alpha, move)
                        is_exact = raw_alpha < score < raw_beta
                    else:
                        is_exact = False
            except SearchTimeout:
                self.undo(token)
                # Rescue a proven better root move if Move #0 already completed and lost out.
                if (
                    rescue_out is not None
                    and index >= 1
                    and best_move is not None
                    and move0_score is not None
                    and best_move != moves[0]
                    and best_score > move0_score
                    and best_score > alpha_orig
                ):
                    rescue_out[:] = [best_move, best_score]
                if best_move is None or depth >= 2 or narrowed:
                    raise
                break
            else:
                self.undo(token)

            if -MATE_BOUND < score < MATE_BOUND:
                score += bias
            if index == 0:
                move0_score = score
            move_scores.append((move, score, is_exact))
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_move is not None and best_score != self.contempt and best_score != -self.contempt:
            if best_score <= alpha_orig:
                flag = UPPER
            elif best_score >= beta:
                flag = LOWER
            else:
                flag = EXACT
            stored = best_score
            if stored > MATE_BOUND:
                stored += self.ply
            elif stored < -MATE_BOUND:
                stored -= self.ply
            previous = self.tt.get(self.zhash)
            if previous is None or depth >= previous[0]:
                store_move = (
                    previous[3]
                    if (flag == UPPER and previous is not None and previous[3] >= 0)
                    else best_move
                )
                if len(self.tt) > TT_LIMIT:
                    self._age_tt()
                self.tt[self.zhash] = (depth, stored, flag, store_move)
        return best_move, best_score, move_scores

    def choose(self, soft_budget: float, hard_budget: float | None = None, opening_variety: bool = False):
        if len(self.rcache) > DIST_CACHE_LIMIT // 2:
            self.rcache.clear()
        if len(self.dag_cache) > DAG_CACHE_LIMIT // 2:
            self.dag_cache.clear()
        if len(self.dcache) > DIST_CACHE_LIMIT // 2:
            self.dcache.clear()
        if len(self.tcache) > THREAT_CACHE_LIMIT // 2:
            self.tcache.clear()

        if hard_budget is None:
            hard_budget = soft_budget
        start = time.perf_counter()
        self.deadline = start + max(0.01, hard_budget)
        self.nodes = 0
        self.ext = 0
        self.line = {self.zhash}
        self.killers = [[-1, -1] for _ in range(96)]
        history = self.history
        for i in range(len(history)):
            history[i] >>= 1

        win = self._winning_destination()
        if win is not None:
            self.completed_depth = 0
            self.completed_score = MATE
            return win

        best_move = None
        best_score = -INF
        self.completed_depth = 0
        self.completed_score = -INF
        completed_move_scores: list[tuple[int, int, bool]] = []
        hopeful = None
        hopeful_depth = 0

        my_dag = self._dag(self.pawns[0], self.goals[0])
        step = self._best_first_step(my_dag[2], self.goals[0])
        if step >= 0 and not self._is_banned(step):
            best_move = step

        stable_iters = 0
        prev_best_move = None
        prev_best_score = None
        loss_cap = min(soft_budget, 0.5)

        for depth in range(1, 32):
            elapsed = time.perf_counter() - start
            if time.perf_counter() >= self.deadline:
                break
            if depth > 2:
                unstable = (
                    prev_best_move is not None
                    and (best_move != prev_best_move or (prev_best_score is not None and abs(best_score - prev_best_score) >= 45))
                )
                stop_share = 0.85 if unstable else 0.58
                if elapsed > stop_share * soft_budget:
                    break

            rescue: list = []
            try:
                if depth >= 3 and abs(best_score) < FORCED_RACE:
                    lo_delta = 140
                    hi_delta = 140
                    lo = best_score - lo_delta
                    hi = best_score + hi_delta
                    while True:
                        move, score, move_scores = self.search_root(
                            depth, lo, hi, opening_variety=opening_variety, rescue_out=rescue
                        )
                        if score <= lo and lo > -INF:
                            if lo_delta == 140:
                                lo_delta = 380
                                lo = best_score - lo_delta
                            else:
                                lo = -INF
                        elif score >= hi and hi < INF:
                            if hi_delta == 140:
                                hi_delta = 380
                                hi = best_score + hi_delta
                            else:
                                hi = INF
                        else:
                            break
                else:
                    move, score, move_scores = self.search_root(
                        depth, -INF, INF, opening_variety=opening_variety, rescue_out=rescue
                    )
            except SearchTimeout:
                if rescue:
                    best_move, best_score = rescue[0], rescue[1]
                    self.completed_score = best_score
                break

            if move is None:
                break
            best_move = move
            best_score = score
            completed_move_scores = move_scores
            self.completed_depth = depth
            self.completed_score = score

            if move == prev_best_move and prev_best_score is not None and abs(score - prev_best_score) < 30:
                stable_iters += 1
            else:
                stable_iters = 1
            prev_best_move = move
            prev_best_score = score

            if score > -3 * STEP:
                hopeful = move
                hopeful_depth = depth
            if score >= MATE - 40 or score >= FORCED_RACE:
                break
            if score <= -(MATE - 40):
                if depth >= 3 or (time.perf_counter() - start) >= loss_cap:
                    break
            if depth >= 8 and stable_iters >= 4 and (time.perf_counter() - start) > 0.55 * soft_budget:
                break

        if best_move is None:
            best_move = self._delaying_move(None)
        elif best_score <= -(MATE - 40):
            if self.completed_depth < 2:
                best_move = self._delaying_move(best_move)
        elif best_score <= -FORCED_RACE:
            distinct_scores = {s for _m, s, _ex in completed_move_scores}
            if len(distinct_scores) <= 1:
                if hopeful is not None and hopeful_depth >= self.completed_depth - 1:
                    best_move = hopeful
                else:
                    best_move = self._delaying_move(best_move)

        if (
            opening_variety
            and completed_move_scores
            and -FORCED_RACE < best_score < MATE - 40
        ):
            near_best = [
                m for m, s, ex in completed_move_scores
                if ex and s >= best_score - 15 and s > -FORCED_RACE
            ]
            if best_move not in near_best and best_move is not None:
                near_best.append(best_move)
            if len(near_best) > 1:
                best_move = random.choice(near_best)

        if best_move is None:
            dests = self._pawn_destinations()
            free = [dest for dest in dests if not self._is_banned(dest)]
            best_move = (free or dests or [0])[0]
        return best_move

    def _delaying_move(self, fallback):
        """Every line loses: delay mate as long as possible without hoarding walls or oscillating."""
        best, best_key = fallback, None
        self.deadline = max(self.deadline, time.perf_counter() + 0.05)
        try:
            moves = self.root_moves(1, -1)
        except SearchTimeout:
            return fallback
        for move in moves:
            token = self.apply(move)
            try:
                if token[1]:
                    return move
                child_hash = self.zhash
                mine, theirs = self.distances()
            finally:
                self.undo(token)
            if mine is None or theirs is None:
                continue
            walls = self.walls_left[0] - (move >= 81)
            repeats = self.played.get(child_hash, 0)
            score = (
                -repeats * 50_000
                + (theirs - mine) * STEP
                + min(theirs, 5) * 200
                - (5_000 if theirs <= 1 else 0)
                - mine * 20
                + walls * 10
            )
            key = (score, theirs, -mine, move < 81)
            if best_key is None or key > best_key:
                best, best_key = move, key
        return best

    def to_action(self, move: int) -> StatelessAction:
        if move < 81:
            row, col = divmod(move, 9)
            return StatelessAction({"type": "move", "destination": (row, col)})
        if move < V_BASE:
            idx = move - H_BASE
            return StatelessAction({
                "type": "horizontal",
                "destination": (idx // 8 + 1, idx & 7),
            })
        idx = move - V_BASE
        return StatelessAction({
            "type": "vertical",
            "destination": (idx // 8, (idx & 7) + 1),
        })

    def move_name(self, move: int) -> str:
        action = self.to_action(move)
        kind = action.data["type"]
        row, col = action.data["destination"]
        if kind == "move":
            return f"move {row},{col}"
        return f"{kind[0].upper()} {row},{col}"


class MyPlayer(PlayerQuoridor):
    """
    Bitboard PVS + Tactical Quiescence Quoridor agent.

    Uses stride-9 bitboard DAG bridge/pair wall-cut detection, staged move
    generation (advancing pawn steps -> unreduced top tactical/synergy/defensive
    walls -> non-advancing steps & setup walls), bounded tactical quiescence for
    goal-doorstep evasions and jump wars, parity-aligned tempo, and soft/hard
    time allocation with partial-root timeout rescue.
    """

    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name, *args, **kwargs)
        self._dag_cache: dict = {}
        self._tt: dict = {}
        self._history = [0] * MOVE_LIMIT
        self._countermoves = [-1] * MOVE_LIMIT
        self._engine = Engine(self._dag_cache, self._tt, self._history, self._countermoves)
        self.last_depth = 0
        self.last_score = 0
        self.last_nodes = 0
        self.last_move = ""
        self._visits: dict[int, int] = {}
        self._last_step = -1
        self._last_contempt: int | None = None
        self._last_banned: frozenset[int] = frozenset()
        self._clock_total = 15 * 60.0
        self._returned_at: float | None = None
        self._opp_used = 0.0
        self._opp_moves = 0
        self._log_path: str | None = None

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15 * 60, **kwargs) -> Action:
        started = time.perf_counter()
        step = current_state.get_step()
        if step <= self._last_step or self._last_step < 0:
            self._new_game(remaining_time)
        elif self._returned_at is not None:
            self._opp_used += started - self._returned_at
            self._opp_moves += 1
        self._last_step = step

        engine = self._engine
        engine.load(current_state, self)
        visits = self._visits.get(engine.zhash, 0) + 1
        self._visits[engine.zhash] = visits
        engine.played = self._visits
        engine.contempt = self._contempt(visits, remaining_time)
        if visits >= LOOP_BAN_VISITS:
            engine.banned = {key for key, count in self._visits.items() if count >= 2}
        else:
            engine.banned = set()

        banned_key = frozenset(engine.banned)
        if self._last_contempt is not None and (
            engine.contempt != self._last_contempt or banned_key != self._last_banned
        ):
            self._tt.clear()
        self._last_contempt = engine.contempt
        self._last_banned = banned_key

        d0, d1 = engine.distances()
        my_d = d0 or 8
        opp_d = d1 or 8
        my_walls, opp_walls = engine.walls_left
        opening = not current_state.rep.walls and min(my_d, opp_d) >= 5
        opening_variety = step <= 6 and len(current_state.rep.walls) == 0 and visits == 1
        contact = engine._pawns_facing()
        pawns_interact = engine._pawns_interact(my_d, opp_d)
        soft_budget, hard_budget = self._budget(
            remaining_time, my_d, opp_d, my_walls, opp_walls,
            opening, step, contact, pawns_interact,
        )
        if visits >= 2:
            soft_budget = min(soft_budget, REPEAT_BUDGET)
            hard_budget = min(hard_budget, REPEAT_BUDGET)

        try:
            move = engine.choose(soft_budget, hard_budget, opening_variety=opening_variety)
            child = engine._child_hash(move)
            self._visits[child] = self._visits.get(child, 0) + 1
            action = engine.to_action(move)
        except Exception:
            engine.load(current_state, self)
            action = None
        if action is None or not self._accepted(current_state, action):
            action = self._fallback(current_state)
            self.last_move = "fallback " + str(action.data)
        else:
            self.last_move = engine.move_name(move)

        self.last_depth = engine.completed_depth
        self.last_score = engine.completed_score
        self.last_nodes = engine.nodes
        self._log(step, visits, remaining_time, soft_budget, started)
        self._returned_at = time.perf_counter()
        return action

    def _new_game(self, remaining: float) -> None:
        self._visits.clear()
        self._tt.clear()
        self._dag_cache.clear()
        self._engine.rcache.clear()
        self._engine.dcache.clear()
        self._engine.tcache.clear()
        self._history = [0] * MOVE_LIMIT
        self._countermoves = [-1] * MOVE_LIMIT
        self._engine.history = self._history
        self._engine.countermoves = self._countermoves
        self._last_contempt = None
        self._last_banned = frozenset()
        self._clock_total = max(remaining, 1.0)
        self._returned_at = None
        self._opp_used = 0.0
        self._opp_moves = 0
        self._log_path = None
        if MOVE_LOG:
            try:
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
                os.makedirs(folder, exist_ok=True)
                name = f"{time.strftime('%Y%m%d-%H%M%S')}-{self.piece_type}-{os.getpid()}.jsonl"
                self._log_path = os.path.join(folder, name)
            except Exception:
                self._log_path = None

    def _opp_clock(self) -> float:
        return self._clock_total - self._opp_used

    def _contempt(self, visits: int, remaining: float) -> int:
        """Price of a shuffle: who runs out first, and how often we have been here."""
        opp_rate = self._opp_used / self._opp_moves if self._opp_moves else 1.0
        ours = remaining / REPEAT_BUDGET
        theirs = self._opp_clock() / max(opp_rate, 0.005)
        base = CONTEMPT_BEHIND_CLOCK if ours <= theirs else CONTEMPT
        return min(CONTEMPT_CAP, base + CONTEMPT_PER_VISIT * (visits - 1))

    def _log(self, step: int, visits: int, remaining: float, budget: float, started: float) -> None:
        if self._log_path is None:
            return
        engine = self._engine
        try:
            record = {
                "step": step,
                "move": self.last_move,
                "depth": self.last_depth,
                "score": self.last_score,
                "nodes": self.last_nodes,
                "seconds": round(time.perf_counter() - started, 4),
                "budget": round(budget, 3),
                "visits": visits,
                "contempt": engine.contempt,
                "banned": len(engine.banned),
                "walls_left": list(engine.walls_left),
                "remaining_time": round(remaining, 2),
                "opp_clock_est": round(self._opp_clock(), 2),
            }
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            self._log_path = None

    def _budget(
        self,
        remaining: float,
        my_d: int,
        opp_d: int,
        my_walls: int,
        opp_walls: int,
        opening: bool = False,
        step: int = 0,
        contact: bool = False,
        pawns_interact: bool = False,
    ) -> tuple[float, float]:
        if remaining <= 0:
            return 0.02, 0.02
        if my_walls + opp_walls == 0:
            b = min(1.0, max(0.02, remaining * 0.02))
            return b, b
        if opp_walls == 0 and my_d <= opp_d and not pawns_interact:
            b = min(0.3, max(0.02, remaining * 0.005))
            return b, b

        moves_left = max(8, my_d + (my_walls + opp_walls) // 2 + 3)
        sharp = abs(my_d - opp_d) <= 3 and my_walls + opp_walls >= 2
        soft = remaining / moves_left * (1.6 if sharp else 1.05)

        total = self._clock_total
        if sharp:
            cap = total * (CAP_SHARP if my_walls >= 4 else CAP_SHARP_FEW_WALLS)
        else:
            cap = total * CAP_ORDINARY
        if opening:
            cap = min(cap, total * CAP_OPENING)

        if step <= TIME_PLAN_PLY:
            share = TIME_PLAN_SHARE * step / TIME_PLAN_PLY
        else:
            share = min(0.92, TIME_PLAN_SHARE + 0.011 * (step - TIME_PLAN_PLY))
        if total - remaining > total * (share + 0.015):
            cap = min(cap, total * CAP_OVER_PLAN)

        if contact and remaining < total * CONTACT_RESERVE:
            cap = min(cap, remaining / 36)
        if remaining < 90:
            cap = min(cap, 4.0)
        if remaining < 30:
            cap = min(cap, 1.2)
        if remaining < 12:
            cap = min(cap, 0.4)

        reserve = 18.0 if remaining > 60 else remaining * 0.25
        avail = max(0.03, remaining - reserve)
        soft = max(0.03, min(soft, cap, avail))
        hard = max(soft, min(soft * 2.1, cap * 1.45, avail))
        if opening:
            hard = min(hard, soft * 1.35)
        return soft, hard

    def _accepted(self, state: GameStateQuoridor, action: StatelessAction) -> bool:
        kind = action.data["type"]
        if kind == "move":
            destination = action.data["destination"]
            return any(move.data["destination"] == destination for move in state._legal_moves())
        row, col = action.data["destination"]
        orientation = Orientation.VERTICAL if kind == "vertical" else Orientation.HORIZONTAL
        return state._is_wall_legal(Wall(row, col, orientation))

    def _fallback(self, state: GameStateQuoridor) -> StatelessAction:
        moves = list(state._legal_moves())
        if not moves:
            return next(iter(state.generate_possible_stateless_actions()))
        opponent = state._opponent(self)
        best = moves[0]
        best_key = -INF
        for move in moves:
            nxt = state.apply_action(move)
            mine = nxt._shortest_path(self)
            theirs = nxt._shortest_path(opponent)
            if mine is None:
                continue
            key = (theirs if theirs is not None else 0) - mine
            if key > best_key:
                best_key = key
                best = move
        return best
