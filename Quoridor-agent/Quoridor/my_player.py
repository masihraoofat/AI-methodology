"""Quoridor agent: iterative-deepening principal-variation search on a bitboard.

Cells are bits of one integer (bit ``r * 9 + c``). Two masks hold the open
edges: ``vmask`` bit ``i`` means ``i <-> i + 9`` is open, ``hmask`` bit ``i``
means ``i <-> i + 1`` is open. A breadth-first layer is then four shifts and
masks, so path lengths cost a handful of integer operations per step.
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

STEP = 120

# ---- Wall pricing -------------------------------------------------------
# Worth of the walls still in hand, indexed by count. Every wall is priced
# above a single pawn step (120) so early walls are not dumped lightly, while
# late walls stay below 2.1x a step to avoid hoarding in losing endgames.
# Marginals, tenth wall to first: 135 145 155 165 175 185 200 215 230 245.
WALL_VALUE = (0, 245, 475, 690, 890, 1075, 1250, 1415, 1570, 1715, 1850)
# Side-to-move tempo bonus.
TEMPO = 25
# A root pawn move that lengthens our own path must beat the others by this much.
REGRESS_PENALTY = 45

# ---- Repetition ---------------------------------------------------------
# The harness has no draw rule: a shuffle ends only when a clock runs out.
# Base dislike of a repeated position while our clock is ahead of theirs.
CONTEMPT = 80
# Our clock is behind theirs: every shuffle ends in our time loss.
CONTEMPT_BEHIND_CLOCK = 3 * STEP
# Added per earlier visit of the root position, so a loop gets more expensive
# each time round until a worse but progressing move wins.
CONTEMPT_PER_VISIT = STEP
CONTEMPT_CAP = 20_000
# Visits of the root position after which moves into twice-seen positions are
# banned outright, forced losses included: a time loss is a loss too.
LOOP_BAN_VISITS = 4
# Budget for a position we have already searched.
REPEAT_BUDGET = 0.2

# ---- Time ---------------------------------------------------------------
# Shares of the whole clock, written against the 15-minute game.
# Share of the clock we may have used by ply TIME_PLAN_PLY.
TIME_PLAN_SHARE = 0.35
TIME_PLAN_PLY = 40
CAP_SHARP = 25 / 900
CAP_SHARP_FEW_WALLS = 16 / 900
CAP_ORDINARY = 12 / 900
CAP_OPENING = 6 / 900
CAP_OVER_PLAN = 9 / 900
# Kept back until the pawns have passed each other.
CONTACT_RESERVE = 150 / 900
# Per step beyond the first that the best single wall would add to a path.
# STRIKE when the wall's owner moves next, DODGE when the victim does.
THREAT_STRIKE = 45
THREAT_DODGE = 25
THREAT_CAP = 5
TT_LIMIT = 1_200_000
# One JSON line per move in logs/ next to this file (opt-in via env var).
MOVE_LOG = os.environ.get("QUORIDOR_MOVE_LOG", "0") == "1"

FULL_V = (1 << 72) - 1
FULL_H = sum(1 << (r * 9 + c) for r in range(9) for c in range(8))
GOAL_MASK = tuple(sum(1 << (r * 9 + c) for c in range(9)) for r in range(9))
ROW = [i // 9 for i in range(81)]
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
    horizontal = [rnd() for _ in range(64)]
    vertical = [rnd() for _ in range(64)]
    walls_left = [[rnd() for _ in range(11)] for _ in range(2)]
    side = rnd()
    goal = rnd()
    return pawns, horizontal, vertical, walls_left, side, goal


Z_PAWN, Z_H, Z_V, Z_LEFT, Z_SIDE, Z_GOAL = _zobrist_tables()


def _edge_walls():
    """Walls that close each directed step, matching ``_candidate_blocking_walls``."""
    table = {}
    v_walls: list[tuple[int, ...]] = [() for _ in range(72)]
    h_walls: list[tuple[int, ...]] = [() for _ in range(81)]
    for r in range(9):
        for c in range(9):
            a = r * 9 + c
            if r < 8:
                walls = []
                if c > 0:
                    walls.append(H_BASE + r * 8 + (c - 1))
                if c < 8:
                    walls.append(H_BASE + r * 8 + c)
                tup = tuple(walls)
                table[(a, a + 9)] = tup
                v_walls[a] = tup
            if r > 0:
                walls = []
                if c > 0:
                    walls.append(H_BASE + (r - 1) * 8 + (c - 1))
                if c < 8:
                    walls.append(H_BASE + (r - 1) * 8 + c)
                table[(a, a - 9)] = tuple(walls)
            if c < 8:
                walls = []
                if r > 0:
                    walls.append(V_BASE + (r - 1) * 8 + c)
                if r < 8:
                    walls.append(V_BASE + r * 8 + c)
                tup = tuple(walls)
                table[(a, a + 1)] = tup
                h_walls[a] = tup
            if c > 0:
                walls = []
                if r > 0:
                    walls.append(V_BASE + (r - 1) * 8 + (c - 1))
                if r < 8:
                    walls.append(V_BASE + r * 8 + (c - 1))
                table[(a, a - 1)] = tuple(walls)
    return table, tuple(v_walls), tuple(h_walls)


EDGE_WALLS, V_EDGE_WALLS, H_EDGE_WALLS = _edge_walls()


def _wall_geometry_tables():
    """Precomputed wall anchor positions, edge flags, touch masks, and conflict sets."""
    pos: dict[int, tuple[int, int]] = {}
    on_edge: dict[int, bool] = {}
    touch_h: dict[int, int] = {}
    touch_v: dict[int, int] = {}
    conflicts: dict[int, tuple[int, ...]] = {}

    for idx in range(64):
        r, c = divmod(idx, 8)
        # Horizontal wall: grid segment y = r + 1, x in [c, c + 2].
        hw = H_BASE + idx
        pos[hw] = (r, c)
        on_edge[hw] = (c == 0 or c == 7)
        th = 0
        if c >= 2:
            th |= 1 << (idx - 2)
        if c <= 5:
            th |= 1 << (idx + 2)
        tv = 0
        for vr in range(max(0, r - 1), min(7, r + 1) + 1):
            for vc in range(max(0, c - 1), min(7, c + 1) + 1):
                if vr == r and vc == c:
                    continue
                tv |= 1 << (vr * 8 + vc)
        touch_h[hw] = th
        touch_v[hw] = tv
        conf = [V_BASE + idx]
        if c > 0:
            conf.append(H_BASE + idx - 1)
        if c < 7:
            conf.append(H_BASE + idx + 1)
        conflicts[hw] = tuple(conf)

        # Vertical wall: grid segment x = c + 1, y in [r, r + 2].
        vw = V_BASE + idx
        pos[vw] = (r, c)
        on_edge[vw] = (r == 0 or r == 7)
        tv2 = 0
        if r >= 2:
            tv2 |= 1 << (idx - 16)
        if r <= 5:
            tv2 |= 1 << (idx + 16)
        th2 = 0
        for wr in range(max(0, r - 1), min(7, r + 1) + 1):
            for wc in range(max(0, c - 1), min(7, c + 1) + 1):
                if wr == r and wc == c:
                    continue
                th2 |= 1 << (wr * 8 + wc)
        touch_h[vw] = th2
        touch_v[vw] = tv2
        conf2 = [H_BASE + idx]
        if r > 0:
            conf2.append(V_BASE + idx - 8)
        if r < 7:
            conf2.append(V_BASE + idx + 8)
        conflicts[vw] = tuple(conf2)

    return pos, on_edge, touch_h, touch_v, conflicts


WALL_POS, WALL_ON_EDGE, WALL_TOUCH_H, WALL_TOUCH_V, WALL_CONFLICTS = _wall_geometry_tables()


def _wall_worth(walls: int, enemy_d: int) -> int:
    """Wall reserve value with a symmetric endgame usability taper when enemy_d <= 2."""
    if enemy_d <= 2 and walls > 2:
        usable = WALL_VALUE[2]
        return usable + ((WALL_VALUE[walls] - usable) >> 2)
    return WALL_VALUE[walls]


class SearchTimeout(Exception):
    """Raised when the move deadline is reached."""


class Engine:
    """Mutable board used only inside the search. ``load`` rebuilds it each turn."""

    __slots__ = (
        "vmask", "hmask", "h_walls", "v_walls", "pawns", "goals",
        "walls_left", "side", "ply", "zhash", "paths", "tt", "history",
        "killers", "deadline", "nodes", "ext", "dcache", "root_pawn",
        "completed_depth", "completed_score", "line", "played", "contempt", "banned",
    )

    def __init__(self, paths: dict, tt: dict, history: list[int]) -> None:
        self.paths = paths
        self.tt = tt
        self.history = history
        self.dcache = {}
        self.vmask = FULL_V
        self.hmask = FULL_H
        self.h_walls = 0
        self.v_walls = 0
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
        # Positions (opponent to move) the root may not move into.
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
        self.h_walls = 0
        self.v_walls = 0
        self.vmask = FULL_V
        self.hmask = FULL_H
        self.side = 0
        self.ply = 0
        self.ext = 0
        for wall in state.rep.walls:
            if wall.orientation == Orientation.HORIZONTAL:
                if 1 <= wall.row <= 8 and 0 <= wall.col <= 7:
                    idx = (wall.row - 1) * 8 + wall.col
                    self.h_walls |= 1 << idx
                    self.vmask &= ~H_CLR[idx]
            elif 0 <= wall.row <= 7 and 1 <= wall.col <= 8:
                idx = wall.row * 8 + (wall.col - 1)
                self.v_walls |= 1 << idx
                self.hmask &= ~V_CLR[idx]
        self.zhash = self._compute_hash()

    def _compute_hash(self) -> int:
        h = Z_PAWN[0][self.pawns[0]] ^ Z_PAWN[1][self.pawns[1]]
        h ^= Z_LEFT[0][self.walls_left[0]] ^ Z_LEFT[1][self.walls_left[1]]
        if self.goals[0] == 8:
            h ^= Z_GOAL
        if self.side:
            h ^= Z_SIDE
        bits = self.h_walls
        idx = 0
        while bits:
            if bits & 1:
                h ^= Z_H[idx]
            bits >>= 1
            idx += 1
        bits = self.v_walls
        idx = 0
        while bits:
            if bits & 1:
                h ^= Z_V[idx]
            bits >>= 1
            idx += 1
        return h

    def geometry_ok(self, move: int) -> bool:
        if move < V_BASE:
            idx = move - H_BASE
            h = self.h_walls
            if (h >> idx) & 1:
                return False
            col = idx & 7
            if col and (h >> (idx - 1)) & 1:
                return False
            if col < 7 and (h >> (idx + 1)) & 1:
                return False
            return ((self.v_walls >> idx) & 1) == 0
        idx = move - V_BASE
        v = self.v_walls
        if (v >> idx) & 1:
            return False
        row = idx >> 3
        if row and (v >> (idx - 8)) & 1:
            return False
        if row < 7 and (v >> (idx + 8)) & 1:
            return False
        return ((self.h_walls >> idx) & 1) == 0

    def _masks_after(self, wall: int) -> tuple[int, int]:
        if wall < V_BASE:
            return self.vmask & ~H_CLR[wall - H_BASE], self.hmask
        return self.vmask, self.hmask & ~V_CLR[wall - V_BASE]

    def apply(self, move: int):
        snap = (
            self.pawns[0], self.pawns[1], self.h_walls, self.v_walls, self.vmask, self.hmask,
            self.walls_left[0], self.walls_left[1], self.side, self.zhash, self.ply,
        )
        side = self.side
        won = False
        if move < 81:
            self.zhash ^= Z_PAWN[side][self.pawns[side]] ^ Z_PAWN[side][move]
            self.pawns[side] = move
            won = ROW[move] == self.goals[side]
        elif move < V_BASE:
            idx = move - H_BASE
            left = self.walls_left[side]
            self.zhash ^= Z_H[idx] ^ Z_LEFT[side][left] ^ Z_LEFT[side][left - 1]
            self.walls_left[side] = left - 1
            self.h_walls |= 1 << idx
            self.vmask &= ~H_CLR[idx]
        else:
            idx = move - V_BASE
            left = self.walls_left[side]
            self.zhash ^= Z_V[idx] ^ Z_LEFT[side][left] ^ Z_LEFT[side][left - 1]
            self.walls_left[side] = left - 1
            self.v_walls |= 1 << idx
            self.hmask &= ~V_CLR[idx]
        self.zhash ^= Z_SIDE
        self.side ^= 1
        self.ply += 1
        return snap, won

    def undo(self, token) -> None:
        (
            self.pawns[0], self.pawns[1], self.h_walls, self.v_walls, self.vmask, self.hmask,
            self.walls_left[0], self.walls_left[1], self.side, self.zhash, self.ply,
        ) = token[0]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _dist(self, start: int, goal_row: int, v: int, h: int):
        """Pawn-ignoring distance on the given edge masks, ``None`` if unreachable."""
        key = (start, goal_row, v, h)
        dcache = self.dcache
        hit = dcache.get(key, -2)
        if hit != -2:
            return hit
        goal = GOAL_MASK[goal_row]
        reached = 1 << start
        d = 0
        while not reached & goal:
            grown = (reached | ((reached >> 9) & v) | ((reached & v) << 9)
                     | ((reached >> 1) & h) | ((reached & h) << 1))
            if grown == reached:
                d = None
                break
            reached = grown
            d += 1
        if len(dcache) < 1_000_000:
            dcache[key] = d
        return d

    def _race_masks(self, start: int, goal_row: int, v: int, h: int):
        """Pawn-ignoring (distance, narrowest layer of the shortest-path DAG) on given masks.

        ``(None, 0)`` when the row is unreachable. A narrowest layer of one cell
        means every shortest path squeezes through that cell.
        """
        key = (start, goal_row + 16, v, h)
        dcache = self.dcache
        hit = dcache.get(key)
        if hit is not None:
            return hit
        goal = GOAL_MASK[goal_row]
        front = 1 << start
        if front & goal:
            result = (0, 9)
        else:
            layers = [front]
            seen = front
            result = (None, 0)
            while True:
                front = (((front >> 9) & v) | ((front & v) << 9)
                         | ((front >> 1) & h) | ((front & h) << 1)) & ~seen
                if not front:
                    break
                back = front & goal
                if back:
                    narrow = _popcount(back)
                    k = len(layers) - 1
                    while k > 0 and narrow > 1:
                        back = layers[k] & (((back >> 9) & v) | ((back & v) << 9)
                                            | ((back >> 1) & h) | ((back & h) << 1))
                        width = _popcount(back)
                        if width < narrow:
                            narrow = width
                        k -= 1
                    result = (len(layers), narrow)
                    break
                layers.append(front)
                seen |= front
        if len(dcache) < 1_000_000:
            dcache[key] = result
        return result

    def _race(self, start: int, goal_row: int):
        """Pawn-ignoring (distance, narrowest layer of the shortest-path DAG)."""
        return self._race_masks(start, goal_row, self.vmask, self.hmask)

    def _path(self, start: int, goal_row: int):
        """Full shortest-path DAG: length, narrowest layer, first step, walls cutting any DAG edge.

        Blockers run from the goal end back to the pawn.
        """
        if ROW[start] == goal_row:
            return (0, 9, -1, ())
        v, h = self.vmask, self.hmask
        key = (start, goal_row, v, h)
        paths = self.paths
        hit = paths.get(key)
        if hit is not None:
            return hit
        result = self._trace(start, goal_row, v, h)
        if len(paths) < 500_000:
            paths[key] = result
        return result

    def _trace(self, start: int, goal_row: int, v: int, h: int):
        goal = GOAL_MASK[goal_row]
        front = 1 << start
        layers = [front]
        seen = front
        while True:
            front = (((front >> 9) & v) | ((front & v) << 9)
                     | ((front >> 1) & h) | ((front & h) << 1)) & ~seen
            if not front:
                return (None, 0, -1, ())
            if front & goal:
                break
            layers.append(front)
            seen |= front
        upper = front & goal
        narrow = _popcount(upper)
        blockers: list[int] = []
        seen_walls: set[int] = set()
        first_step = -1
        for k in range(len(layers) - 1, -1, -1):
            if k > 0:
                lower = layers[k] & (((upper >> 9) & v) | ((upper & v) << 9)
                                     | ((upper >> 1) & h) | ((upper & h) << 1))
                width = _popcount(lower)
                if width < narrow:
                    narrow = width
            else:
                lower = layers[0]
                bits = upper
                best_step = -1
                best_key = -1000
                while bits:
                    lsb = bits & -bits
                    bits ^= lsb
                    cell = lsb.bit_length() - 1
                    key = -abs(ROW[cell] - goal_row) * 10 - abs((cell % 9) - 4)
                    if key > best_key:
                        best_key = key
                        best_step = cell
                first_step = best_step

            v_edges = v & ((lower & (upper >> 9)) | (upper & (lower >> 9)))
            while v_edges:
                lsb = v_edges & -v_edges
                v_edges ^= lsb
                for wall in V_EDGE_WALLS[lsb.bit_length() - 1]:
                    if wall not in seen_walls:
                        seen_walls.add(wall)
                        blockers.append(wall)

            h_edges = h & ((lower & (upper >> 1)) | (upper & (lower >> 1)))
            while h_edges:
                lsb = h_edges & -h_edges
                h_edges ^= lsb
                for wall in H_EDGE_WALLS[lsb.bit_length() - 1]:
                    if wall not in seen_walls:
                        seen_walls.add(wall)
                        blockers.append(wall)

            upper = lower
        return (len(layers), narrow, first_step, tuple(blockers))

    def _single_blockers(self, start: int, goal_row: int, v: int, h: int) -> frozenset[int]:
        """Walls cutting one shortest path from start to goal_row (superset of all lengthening walls)."""
        if ROW[start] == goal_row:
            return frozenset()
        key = (start, goal_row + 48, v, h)
        dcache = self.dcache
        hit = dcache.get(key)
        if hit is not None:
            return hit
        goal = GOAL_MASK[goal_row]
        front = 1 << start
        layers = [front]
        seen = front
        while True:
            front = (((front >> 9) & v) | ((front & v) << 9)
                     | ((front >> 1) & h) | ((front & h) << 1)) & ~seen
            if not front:
                res = frozenset()
                if len(dcache) < 1_000_000:
                    dcache[key] = res
                return res
            if front & goal:
                break
            layers.append(front)
            seen |= front
        cur = (front & goal) & -(front & goal)
        cell = cur.bit_length() - 1
        walls: set[int] = set()
        for k in range(len(layers) - 1, 0, -1):
            step = layers[k] & (((cur >> 9) & v) | ((cur & v) << 9)
                                | ((cur >> 1) & h) | ((cur & h) << 1))
            nxt = step & -step
            ncell = nxt.bit_length() - 1
            walls.update(EDGE_WALLS[(ncell, cell)])
            cur = nxt
            cell = ncell
        walls.update(EDGE_WALLS[(start, cell)])
        res = frozenset(walls)
        if len(dcache) < 1_000_000:
            dcache[key] = res
        return res

    def _raw_threats(self, start: int, goal_row: int, d: int, v: int, h: int) -> tuple[tuple[int, int], ...]:
        """Cached list of (delta_d, wall) for geometrically legal walls that lengthen this path."""
        key = (start, goal_row + 32, v, h)
        dcache = self.dcache
        hit = dcache.get(key)
        if hit is not None:
            return hit
        raw: list[tuple[int, int]] = []
        if d:
            dist = self._dist
            for wall in self._single_blockers(start, goal_row, v, h):
                if not self.geometry_ok(wall):
                    continue
                nv, nh = self._masks_after(wall)
                nd = dist(start, goal_row, nv, nh)
                if nd is not None and nd > d:
                    raw.append((nd - d, wall))
            if len(raw) > 1:
                raw.sort(reverse=True)
        res = tuple(raw)
        if len(dcache) < 1_000_000:
            dcache[key] = res
        return res

    def _threat(
        self,
        defender: int,
        defender_goal: int,
        defender_d: int,
        attacker: int,
        attacker_goal: int,
        attacker_d: int,
    ) -> int:
        """Maximum legal net detour one wall can inflict on defender vs attacker."""
        if not defender_d:
            return 0
        v, h = self.vmask, self.hmask
        raw = self._raw_threats(defender, defender_goal, defender_d, v, h)
        if not raw:
            return 0
        worst = 0
        dist = self._dist
        attacker_single: frozenset[int] | None = None
        for delta_def, wall in raw:
            if delta_def <= worst:
                break
            if attacker_single is None:
                attacker_single = self._single_blockers(attacker, attacker_goal, v, h)
            if wall in attacker_single:
                nv, nh = self._masks_after(wall)
                na = dist(attacker, attacker_goal, nv, nh)
                if na is None:
                    continue
                net = delta_def - (na - attacker_d)
            else:
                net = delta_def
            if net > worst:
                worst = net
        return worst

    def _cell_dist(self, start: int, target: int):
        """Pawn-ignoring steps between two cells."""
        if start == target:
            return 0
        v, h = self.vmask, self.hmask
        key = (start, -1 - target, v, h)
        dcache = self.dcache
        hit = dcache.get(key)
        if hit is not None:
            return None if hit < 0 else hit
        goal = 1 << target
        front = seen = 1 << start
        d = 0
        found = -1
        while True:
            front = (((front >> 9) & v) | ((front & v) << 9)
                     | ((front >> 1) & h) | ((front & h) << 1)) & ~seen
            if not front:
                break
            d += 1
            if front & goal:
                found = d
                break
            seen |= front
        if len(dcache) < 1_000_000:
            dcache[key] = found
        return None if found < 0 else found

    def _on_shortest_path(self, cell: int, start: int, goal_row: int, path_len: int) -> bool:
        if cell == start:
            return False
        via = self._cell_dist(start, cell)
        if via is None:
            return False
        rest = self._dist(cell, goal_row, self.vmask, self.hmask)
        if rest is None:
            return False
        return via + rest == path_len

    def _pawns_interact(self, d0: int, d1: int) -> bool:
        """True unless the pawns mathematically cannot meet before the leader finishes.

        Uses wall-aware graph distance ``sep = _cell_dist(p0, p1)``. Over at most
        ``leader`` turns (<= 2 * leader - 1 plies before the leader's final step),
        the graph separation between pawns can shrink by at most 2 * leader - 1,
        so if ``sep > 2 * leader`` they can never become adjacent.
        """
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

    def races(self):
        """Both players' (distance, narrowest layer) on the current walls."""
        return self._race(self.pawns[0], self.goals[0]), self._race(self.pawns[1], self.goals[1])

    def distances(self):
        v, h = self.vmask, self.hmask
        return (
            self._dist(self.pawns[0], self.goals[0], v, h),
            self._dist(self.pawns[1], self.goals[1], v, h),
        )

    def analyze(self):
        """Both players: distance, narrowest layer, first step, walls on one shortest path."""
        return self._path(self.pawns[0], self.goals[0]), self._path(self.pawns[1], self.goals[1])

    def _open(self, a: int, b: int) -> bool:
        d = b - a
        if d == 9:
            return (self.vmask >> a) & 1 == 1
        if d == -9:
            return (self.vmask >> b) & 1 == 1
        if d == 1:
            return (self.hmask >> a) & 1 == 1
        return (self.hmask >> b) & 1 == 1

    def _pawn_destinations(self) -> list[int]:
        side = self.side
        start = self.pawns[side]
        r, c = divmod(start, 9)
        found = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            found.extend(self._try_dir(r, c, dr, dc))
        if len(found) < 2:
            return found
        unique = []
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
            straight_open = (
                0 <= jr < 9 and 0 <= jc < 9 and self._open(b, jr * 9 + jc)
            )
            if straight_open:
                return self._try_dir(nr, nc, dr, dc)
            if dr == 0:
                return self._try_dir(nr, nc, -1, 0) + self._try_dir(nr, nc, 1, 0)
            return self._try_dir(nr, nc, 0, -1) + self._try_dir(nr, nc, 0, 1)
        return [b]

    def _winning_destination(self) -> int | None:
        goal = self.goals[self.side]
        if abs(self.pawns[self.side] // 9 - goal) > 2:
            return None
        for dest in self._pawn_destinations():
            if dest // 9 == goal:
                return dest
        return None

    # ------------------------------------------------------------------
    # Evaluation and moves
    # ------------------------------------------------------------------

    def evaluate(self, alpha: int = -INF, beta: int = INF) -> int:
        if self._winning_destination() is not None:
            return MATE - (self.ply + 1)

        side = self.side
        opp = side ^ 1
        r0, r1 = self.races()
        d0, d1 = r0[0], r1[0]
        if side == 0:
            (my_d, my_narrow), (opp_d, opp_narrow) = r0, r1
        else:
            (my_d, my_narrow), (opp_d, opp_narrow) = r1, r0
        if my_d is None:
            return -MATE + self.ply
        if opp_d is None:
            return MATE - self.ply

        my_left = self.walls_left[side]
        opp_left = self.walls_left[opp]
        # Positive when the side to move wins a pure race (equal distance included:
        # they move first).
        adv = opp_d - my_d
        # A static race is finished only while a jump cannot rewrite it.
        # Otherwise keep the normal gradient and let the search play the meeting.
        if opp_left == 0 and adv >= 0 and not self._pawns_interact(d0, d1):
            return FORCED_RACE + 10_000 - my_d * 100 + opp_d
        if my_left == 0 and adv < 0 and not self._pawns_interact(d0, d1):
            return -(FORCED_RACE + 10_000) + opp_d * 100 - my_d

        score = adv * STEP + TEMPO + _wall_worth(my_left, opp_d) - _wall_worth(opp_left, my_d)
        # Symmetric centrality difference.
        score += (abs((self.pawns[opp] % 9) - 4) - abs((self.pawns[side] % 9) - 4)) * 4

        # Immediate jump-ready tempo bonus when pawns are orthogonally adjacent with an open edge.
        my_p = self.pawns[side]
        opp_p = self.pawns[opp]
        diff = my_p - opp_p
        if (
            (diff == 9 or diff == -9 or (diff == 1 and (my_p % 9) != 0) or (diff == -1 and (opp_p % 9) != 0))
            and self._open(my_p, opp_p)
        ):
            my_goal = self.goals[side]
            v, h = self.vmask, self.hmask
            for dest in self._pawn_destinations():
                nd = self._dist(dest, my_goal, v, h)
                if nd is not None and nd <= my_d - 2:
                    score += STEP // 2
                    break

        # Sound lazy evaluation bounds for the expensive _threat() terms.
        check_dodge = bool(opp_left and my_narrow <= 2)
        check_strike = bool(my_left and opp_narrow <= 2)
        max_neg = (THREAT_CAP - 1) * THREAT_DODGE if check_dodge else 0
        max_pos = (THREAT_CAP - 1) * THREAT_STRIKE if check_strike else 0
        if score - max_neg >= beta:
            return score - max_neg
        if score + max_pos <= alpha:
            return score + max_pos

        # A pawn in a corridor one wall can seal is far worse off than its
        # distance says: the holder of that wall plays it at the last moment.
        # The side to move can still step out of a trap or drop its own wall first,
        # so its exposure counts less than the one it can spring right now.
        if check_dodge:
            threat = self._threat(
                my_p, self.goals[side], my_d,
                opp_p, self.goals[opp], opp_d,
            )
            if threat > 1:
                score -= (min(threat, THREAT_CAP) - 1) * THREAT_DODGE
            elif my_narrow <= 1:
                score -= 16
            if score >= beta:
                return score
            if score + max_pos <= alpha:
                return score + max_pos
        if check_strike:
            threat = self._threat(
                opp_p, self.goals[opp], opp_d,
                my_p, self.goals[side], my_d,
            )
            if threat > 1:
                score += (min(threat, THREAT_CAP) - 1) * THREAT_STRIKE
            elif opp_narrow <= 1:
                score += 16
        return score

    def _collect_walls(self, depth: int) -> list[tuple[int, int]]:
        side = self.side
        if self.walls_left[side] <= 0:
            return []
        opp = side ^ 1
        pawns = self.pawns
        goals = self.goals
        my_d, my_narrow = self._race(pawns[side], goals[side])
        opp_d, opp_narrow, _opp_step, blockers = self._path(pawns[opp], goals[opp])
        if my_d is None or opp_d is None or not blockers:
            return []
        # Ahead, they have no walls left, and pawns cannot meet: just run.
        if self.walls_left[opp] == 0 and opp_d >= my_d:
            d0, d1 = (my_d, opp_d) if side == 0 else (opp_d, my_d)
            if not self._pawns_interact(d0, d1):
                return []

        root = self.ply == 0
        critical = opp_d <= 2
        # Only trim mid-path blockers at shallow depths (depth <= 2).
        if not root and not critical and depth <= 2 and len(blockers) > 8:
            keep = list(blockers[:4]) + list(blockers[-6:])
            trimmed = []
            seen_walls = set()
            for wall in keep:
                if wall not in seen_walls:
                    seen_walls.add(wall)
                    trimmed.append(wall)
            blockers = trimmed
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout

        vmask, hmask = self.vmask, self.hmask
        my_blocks = self._single_blockers(pawns[side], goals[side], vmask, hmask)
        opp_r, opp_c = divmod(pawns[opp], 9)
        dist = self._dist
        h_walls = self.h_walls
        v_walls = self.v_walls
        allow_setup = root or depth >= 2
        allow_zero_gain = (root or depth >= 2) and (opp_d <= 5 or opp_narrow <= 2)

        tactical: list[tuple[int, int]] = []
        setup: list[tuple[int, int]] = []

        for wall in blockers:
            if not self.geometry_ok(wall):
                continue
            v, h = self._masks_after(wall)
            new_opp = dist(pawns[opp], goals[opp], v, h)
            if new_opp is None:
                continue
            if wall in my_blocks:
                new_my = dist(pawns[side], goals[side], v, h)
                if new_my is None:
                    continue
            else:
                new_my = my_d
            delta_opp = new_opp - opp_d
            delta_my = new_my - my_d
            gain = delta_opp - delta_my
            hist_bonus = min(self.history[wall], 4_000) >> 4

            wr, wc = WALL_POS[wall]
            dr = 0 if wr <= opp_r <= wr + 1 else (wr - opp_r if opp_r < wr else opp_r - (wr + 1))
            dc = 0 if wc <= opp_c <= wc + 1 else (wc - opp_c if opp_c < wc else opp_c - (wc + 1))
            mdist = dr + dc

            if critical and delta_opp >= 1:
                priority = 1_800_000 + delta_opp * 10_000 - delta_my * 1_000 - mdist * 20 + hist_bonus
                tactical.append((priority, wall))
            elif delta_opp >= 1 and gain >= 1:
                priority = 1_500_000 + gain * 10_000 + delta_opp * 100 - delta_my * 100 - mdist * 20 + hist_bonus
                tactical.append((priority, wall))
            elif delta_opp >= 1 and gain == 0 and allow_zero_gain:
                priority = 1_100_000 + delta_opp * 5_000 - mdist * 20 + hist_bonus
                tactical.append((priority, wall))
            elif allow_setup and delta_opp == 0 and delta_my == 0:
                touches = WALL_ON_EDGE[wall] or bool(
                    (h_walls & WALL_TOUCH_H[wall]) | (v_walls & WALL_TOUCH_V[wall])
                )
                narrows = False
                new_narrow = opp_narrow
                if opp_narrow >= 2:
                    new_narrow = self._race_masks(pawns[opp], goals[opp], v, h)[1]
                    narrows = new_narrow < opp_narrow
                if narrows or (touches and mdist <= 3) or mdist <= 2:
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

        # Defensive / anti-trap walls when our own path is narrow and opponent has walls.
        defensive: list[tuple[int, int]] = []
        if self.walls_left[opp] > 0 and my_narrow <= 2 and allow_setup:
            raw_my = self._raw_threats(pawns[side], goals[side], my_d, vmask, hmask)
            if raw_my:
                opp_single = self._single_blockers(pawns[opp], goals[opp], vmask, hmask)
                threats: list[tuple[int, int]] = []
                for d_my, tw in raw_my:
                    if d_my < 2:
                        break
                    if tw in opp_single:
                        tv, th = self._masks_after(tw)
                        new_opp_t = dist(pawns[opp], goals[opp], tv, th)
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
                        for wall in WALL_CONFLICTS[tw]:
                            if wall in def_seen or not self.geometry_ok(wall):
                                continue
                            def_seen.add(wall)
                            v, h = self._masks_after(wall)
                            if wall in my_blocks:
                                new_my = dist(pawns[side], goals[side], v, h)
                                if new_my != my_d:
                                    continue
                            if wall in opp_single:
                                new_opp = dist(pawns[opp], goals[opp], v, h)
                                if new_opp is None:
                                    continue
                                delta_opp = new_opp - opp_d
                            else:
                                delta_opp = 0
                            prio = (
                                1_200_000
                                + net_t * 10_000
                                + delta_opp * 5_000
                                + (min(self.history[wall], 4_000) >> 4)
                            )
                            defensive.append((prio, wall))

        if not tactical and not defensive and not setup:
            return []
        tactical.sort(reverse=True)
        defensive.sort(reverse=True)
        setup.sort(reverse=True)

        if root:
            tac_limit, def_limit, setup_limit = 14, 3, 3
        elif critical:
            tac_limit, def_limit, setup_limit = 10, 2, 2
        elif depth >= 8:
            tac_limit, def_limit, setup_limit = 7, 3, 3
        elif depth >= 5:
            tac_limit, def_limit, setup_limit = 5, 2, 2
        elif depth >= 3:
            tac_limit, def_limit, setup_limit = 4, 2, 2
        else:
            tac_limit, def_limit, setup_limit = 3, 2, 2

        combined: list[tuple[int, int]] = []
        used: set[int] = set()
        for bucket, lim in ((tactical, tac_limit), (defensive, def_limit), (setup, setup_limit)):
            count = 0
            for item in bucket:
                w = item[1]
                if w in used:
                    continue
                used.add(w)
                combined.append(item)
                count += 1
                if count >= lim:
                    break
        combined.sort(reverse=True)
        return combined

    def _pawn_moves(self, skip_move: int = -1) -> list[int]:
        side = self.side
        goal = self.goals[side]
        v, h = self.vmask, self.hmask
        dist = self._dist
        my_d = dist(self.pawns[side], goal, v, h)
        scored: list[tuple[int, int]] = []
        for dest in self._pawn_destinations():
            if dest == skip_move:
                continue
            if dest // 9 == goal:
                priority = 3_000_000
            else:
                d_dest = dist(dest, goal, v, h)
                if my_d is not None and d_dest is not None and d_dest < my_d:
                    priority = 2_000_000 + (my_d - d_dest) * 100_000 - abs((dest % 9) - 4)
                else:
                    priority = 1_000_000 - abs(ROW[dest] - goal) * 1_000 - abs((dest % 9) - 4)
            priority += min(self.history[dest], 4_000) >> 4
            scored.append((priority, dest))
        if len(scored) > 1:
            scored.sort(reverse=True)
        return [move for _priority, move in scored]

    def generate(self, depth: int, tt_move: int) -> list[int]:
        side = self.side
        goal = self.goals[side]
        my_step = self._path(self.pawns[side], goal)[2]
        scored: list[tuple[int, int]] = []

        for dest in self._pawn_destinations():
            if dest // 9 == goal:
                priority = 3_000_000
            elif dest == my_step:
                priority = 2_000_000
            else:
                # Row progress first; the search, not this sort key, decides jumps.
                priority = 1_000_000 - abs(ROW[dest] - goal) * 1_000 - abs((dest % 9) - 4)
            priority += min(self.history[dest], 4_000) >> 4
            scored.append((priority, dest))

        scored.extend(self._collect_walls(depth))
        scored.sort(reverse=True)
        ordered = [move for _priority, move in scored]

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
                # Only promote wall killers already in context (never inject foreign walls).
                ordered.remove(killer)
                ordered.insert(insert_at, killer)
                insert_at += 1
        return ordered

    def _plausible(self, move: int) -> bool:
        if move < 81:
            return move in self._pawn_destinations()
        if self.walls_left[self.side] <= 0 or not self.geometry_ok(move):
            return False
        v, h = self._masks_after(move)
        return (
            self._dist(self.pawns[0], self.goals[0], v, h) is not None
            and self._dist(self.pawns[1], self.goals[1], v, h) is not None
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def negamax(self, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout

        if self._winning_destination() is not None:
            return MATE - (self.ply + 1)

        zhash = self.zhash
        # Back to a position already on the board or on this line: both pawns
        # are shuffling, which ends only on a clock. ``contempt`` prices it for
        # us, the root side, from the clocks and how often the root recurred.
        if zhash in self.line or zhash in self.played:
            return -self.contempt if self.side == 0 else self.contempt

        extended = False
        opp_d = None
        if 0 < depth < 6 and self.ext < 2 and self.walls_left[self.side] > 0:
            packed = self.distances()
            opp_d = packed[1] if self.side == 0 else packed[0]
            if opp_d is not None and opp_d <= 2:
                depth += 1
                self.ext += 1
                extended = True

        self.line.add(zhash)
        try:
            alpha_orig = alpha
            tt_move = -1
            entry = self.tt.get(self.zhash)
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
                        return tt_score

            if depth <= 0:
                return self.evaluate(alpha, beta)

            best = -INF
            best_move = -1
            tried_tt = -1
            node_ply = self.ply
            move_idx = 0

            # Stage 1: Hash move first. A cutoff here skips all move generation.
            if tt_move >= 0 and self._plausible(tt_move):
                tried_tt = tt_move
                token = self.apply(tt_move)
                try:
                    if token[1]:
                        score = MATE - self.ply
                    else:
                        score = -self.negamax(depth - 1, -beta, -alpha)
                finally:
                    self.undo(token)
                best = score
                best_move = tt_move
                move_idx = 1
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self._push_killer(node_ply, tt_move)
                    self.history[tt_move] += depth * depth

            # Stage 2: Pawn moves before BFS-heavy wall generation.
            if alpha < beta:
                first = best_move < 0
                for move in self._pawn_moves(tried_tt):
                    token = self.apply(move)
                    try:
                        if token[1]:
                            score = MATE - self.ply
                        elif first:
                            score = -self.negamax(depth - 1, -beta, -alpha)
                        else:
                            score = -self.negamax(depth - 1, -alpha - 1, -alpha)
                            if alpha < score < beta:
                                score = -self.negamax(depth - 1, -beta, -alpha)
                    finally:
                        self.undo(token)
                    first = False
                    move_idx += 1
                    if score > best:
                        best = score
                        best_move = move
                    if best > alpha:
                        alpha = best
                    if alpha >= beta:
                        self.history[move] += depth * depth
                        break

            # Stage 3: Wall moves only if pawn moves did not cause a beta cutoff.
            if alpha < beta and self.walls_left[self.side] > 0:
                wall_scored = self._collect_walls(depth)
                if wall_scored:
                    walls = [move for _priority, move in wall_scored]
                    kply = node_ply if node_ply < len(self.killers) else len(self.killers) - 1
                    killers_here = self.killers[kply]
                    insert_at = 0
                    for killer in killers_here:
                        if killer >= 81 and killer != tried_tt and killer in walls:
                            walls.remove(killer)
                            walls.insert(insert_at, killer)
                            insert_at += 1
                    if opp_d is None:
                        opp = self.side ^ 1
                        opp_d = self._dist(self.pawns[opp], self.goals[opp], self.vmask, self.hmask)
                    lmr_ok = depth >= 3 and (opp_d is None or opp_d > 2)
                    first = best_move < 0
                    for move in walls:
                        if move == tried_tt:
                            continue
                        is_killer = move == killers_here[0] or move == killers_here[1]
                        token = self.apply(move)
                        try:
                            if first:
                                score = -self.negamax(depth - 1, -beta, -alpha)
                            elif lmr_ok and move_idx >= 4 and not is_killer:
                                score = -self.negamax(depth - 2, -alpha - 1, -alpha)
                                if score > alpha:
                                    score = -self.negamax(depth - 1, -alpha - 1, -alpha)
                                    if alpha < score < beta:
                                        score = -self.negamax(depth - 1, -beta, -alpha)
                            else:
                                score = -self.negamax(depth - 1, -alpha - 1, -alpha)
                                if alpha < score < beta:
                                    score = -self.negamax(depth - 1, -beta, -alpha)
                        finally:
                            self.undo(token)
                        first = False
                        move_idx += 1
                        if score > best:
                            best = score
                            best_move = move
                        if best > alpha:
                            alpha = best
                        if alpha >= beta:
                            self._push_killer(node_ply, move)
                            self.history[move] += depth * depth
                            break

            if best_move < 0:
                return self.evaluate()

            # Do not store path-dependent repetition/contempt scores into TT.
            if best != self.contempt and best != -self.contempt:
                stored = best
                if stored > MATE_BOUND:
                    stored += self.ply
                elif stored < -MATE_BOUND:
                    stored -= self.ply
                if best <= alpha_orig:
                    flag = UPPER
                elif best >= beta:
                    flag = LOWER
                else:
                    flag = EXACT
                previous = self.tt.get(self.zhash)
                if previous is None or depth >= previous[0]:
                    if len(self.tt) > TT_LIMIT:
                        self._age_tt()
                    self.tt[self.zhash] = (depth, stored, flag, best_move)
            return best
        finally:
            self.line.discard(zhash)
            if extended:
                self.ext -= 1

    def _age_tt(self) -> None:
        """Drop the shallow entries first; the deep ones hold the expensive work."""
        tt = self.tt
        for keep in (3, 5, 8):
            for key in [key for key, entry in tt.items() if entry[0] < keep]:
                del tt[key]
            if len(tt) <= TT_LIMIT // 2:
                return
        tt.clear()

    def _push_killer(self, ply: int, move: int) -> None:
        if ply >= len(self.killers) or move < 81:
            return
        first, second = self.killers[ply]
        if move != first:
            self.killers[ply][0] = move
            self.killers[ply][1] = first if first != move else second

    def _root_bias(self, move: int, my_d: int, opp_d: int | None = None) -> int:
        """Discourage regressive pawn shuffles and isolated far back-rank +1 wall dumps."""
        if move < 81:
            if my_d is None:
                return 0
            after = self._dist(move, self.goals[0], self.vmask, self.hmask)
            if after is not None and after > my_d:
                return -REGRESS_PENALTY
            return 0
        # Penalize isolated floating walls placed far from the opponent on an open board
        # that only add +1 statically (horizon-effect back-rank wall spam).
        if opp_d is not None and opp_d >= 4:
            touches = WALL_ON_EDGE[move] or bool(
                (self.h_walls & WALL_TOUCH_H[move]) | (self.v_walls & WALL_TOUCH_V[move])
            )
            if not touches:
                opp_r, opp_c = divmod(self.pawns[1], 9)
                wr, wc = WALL_POS[move]
                dr = 0 if wr <= opp_r <= wr + 1 else (wr - opp_r if opp_r < wr else opp_r - (wr + 1))
                dc = 0 if wc <= opp_c <= wc + 1 else (wc - opp_c if opp_c < wc else opp_c - (wc + 1))
                mdist = dr + dc
                if mdist >= 3:
                    v, h = self._masks_after(move)
                    new_opp = self._dist(self.pawns[1], self.goals[1], v, h)
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
        """Root candidates without the moves that close a loop, unless nothing else is left."""
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
                    score = -self.negamax(depth - 1, -raw_beta, -raw_alpha)
                    is_exact = raw_alpha < score < raw_beta
                elif opening_variety and -MATE_BOUND < raw_alpha < MATE_BOUND:
                    # Keep exact scores for all moves within 15 points of best_score.
                    band = raw_alpha - 16
                    score = -self.negamax(depth - 1, -band - 1, -band)
                    if score > band:
                        score = -self.negamax(depth - 1, -raw_beta, -band)
                        is_exact = band < score < raw_beta
                    else:
                        is_exact = False
                else:
                    score = -self.negamax(depth - 1, -raw_alpha - 1, -raw_alpha)
                    if raw_alpha < score < raw_beta:
                        score = -self.negamax(depth - 1, -raw_beta, -raw_alpha)
                        is_exact = raw_alpha < score < raw_beta
                    else:
                        is_exact = False
            except SearchTimeout:
                self.undo(token)
                if best_move is None or depth >= 2 or narrowed:
                    raise
                break
            else:
                self.undo(token)
            if -MATE_BOUND < score < MATE_BOUND:
                score += bias
            move_scores.append((move, score, is_exact))
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        # Seed root TT entry so subsequent iterations and re-searches try best_move first.
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
                if len(self.tt) > TT_LIMIT:
                    self._age_tt()
                self.tt[self.zhash] = (depth, stored, flag, best_move)
        return best_move, best_score, move_scores

    def choose(self, budget: float, opening_variety: bool = False):
        if len(self.dcache) > 400_000:
            self.dcache.clear()
        if len(self.paths) > 200_000:
            self.paths.clear()

        start = time.perf_counter()
        self.deadline = start + max(0.01, budget)
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
        # Best move of the deepest iteration that still saw a game left to play.
        hopeful = None
        hopeful_depth = 0
        info = self.analyze()
        step = info[0][2]
        if step >= 0 and not self._is_banned(step):
            best_move = step

        stable_iters = 0
        prev_best_move = None
        prev_best_score = None
        loss_cap = min(budget, 0.5)

        for depth in range(1, 32):
            now = time.perf_counter()
            # The next iteration costs several times the last one; do not start
            # what cannot finish.
            if now >= self.deadline or (depth > 2 and now - start > 0.45 * budget):
                break
            try:
                if depth >= 3 and abs(best_score) < FORCED_RACE:
                    lo_delta = 150
                    hi_delta = 150
                    lo = best_score - lo_delta
                    hi = best_score + hi_delta
                    while True:
                        move, score, move_scores = self.search_root(
                            depth, lo, hi, opening_variety=opening_variety
                        )
                        if score <= lo and lo > -INF:
                            if lo_delta == 150:
                                lo_delta = 400
                                lo = best_score - lo_delta
                            else:
                                lo = -INF
                        elif score >= hi and hi < INF:
                            if hi_delta == 150:
                                hi_delta = 400
                                hi = best_score + hi_delta
                            else:
                                hi = INF
                        else:
                            break
                else:
                    move, score, move_scores = self.search_root(
                        depth, -INF, INF, opening_variety=opening_variety
                    )
            except SearchTimeout:
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
                # Do not stop at depth 1/2 (e.g. shallow TT hit); search through at least
                # depth 3 within loss_cap so all root moves compare distance-to-mate.
                if depth >= 3 or (time.perf_counter() - start) >= loss_cap:
                    break
            if depth >= 5 and stable_iters >= 3 and (time.perf_counter() - start) > 0.35 * budget:
                break

        # Forced-loss handling:
        # When proven mate (<= -(MATE - 40)) at completed_depth >= 2, trust search's
        # best_move (-(MATE - ply) already prefers maximum distance-to-mate).
        # Only invoke _delaying_move() if best_move is None, proven mate had depth < 2,
        # or static FORCED_RACE had all root moves tied at the exact same score.
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

        # Opening variety: pick randomly among sound near-equal top moves (within 15 pts).
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
        # Runs after the search, often past its deadline; wall generation checks it.
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
                r0, r1 = self.races()
            finally:
                self.undo(token)
            mine, theirs = r0[0], r1[0]
            if mine is None or theirs is None:
                continue
            walls = self.walls_left[0] - (move >= 81)
            repeats = self.played.get(child_hash, 0)
            # Penalize repeats heavily; prioritize delaying opponent (extra urgency when
            # theirs <= 4); tie-break by advancing our pawn and tiny credit for walls.
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
    Search player.

    Each turn is an iterative-deepening principal-variation search. Walls are
    generated only where they touch the opponent's shortest path, which is
    enough to find every wall that actually lengthens that path. The score is
    the race (their distance minus ours, with the side to move keeping the
    tempo) plus the worth of the walls each side still holds and a corridor term.
    """

    def __init__(self, piece_type: str, goal_row: int = 0, name: str = "bob", *args, **kwargs) -> None:
        super().__init__(piece_type, goal_row, name, *args, **kwargs)
        self._paths: dict = {}
        self._tt: dict = {}
        self._history = [0] * 264
        self._engine = Engine(self._paths, self._tt, self._history)
        self.last_depth = 0
        self.last_score = 0
        self.last_nodes = 0
        self.last_move = ""
        # Visits of every position this game, both sides to move. Kept across
        # walls: the hash holds the walls, so old entries simply never match.
        self._visits: dict[int, int] = {}
        self._last_step = -1
        self._last_contempt: int | None = None
        self._last_banned: frozenset[int] = frozenset()
        self._clock_total = 15 * 60.0
        # The harness only tells us our own clock. Theirs is the wall time
        # between our replies, transport included.
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

        info = engine.analyze()
        my_d = info[0][0] or 8
        opp_d = info[1][0] or 8
        my_walls, opp_walls = engine.walls_left
        opening = not current_state.rep.walls and min(my_d, opp_d) >= 5
        opening_variety = step <= 6 and len(current_state.rep.walls) == 0 and visits == 1
        contact = engine._pawns_facing()
        pawns_interact = engine._pawns_interact(my_d, opp_d)
        budget = self._budget(
            remaining_time, my_d, opp_d, my_walls, opp_walls,
            opening, step, contact, pawns_interact,
        )
        if visits >= 2:
            # Searched before; cap repeat think time.
            budget = min(budget, REPEAT_BUDGET)

        try:
            move = engine.choose(budget, opening_variety=opening_variety)
            child = engine._child_hash(move)
            self._visits[child] = self._visits.get(child, 0) + 1
            action = engine.to_action(move)
        except Exception:
            # An exception out of compute_action is a disqualification.
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
        self._log(step, visits, remaining_time, budget, started)
        self._returned_at = time.perf_counter()
        return action

    def _new_game(self, remaining: float) -> None:
        self._visits.clear()
        self._tt.clear()
        self._paths.clear()
        self._engine.dcache.clear()
        self._history = [0] * 264
        self._engine.history = self._history
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
        # Moves each side can still afford while shuffling.
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

    def _budget(self, remaining: float, my_d: int, opp_d: int, my_walls: int, opp_walls: int,
                opening: bool = False, step: int = 0, contact: bool = False,
                pawns_interact: bool = False) -> float:
        if remaining <= 0:
            return 0.02
        # Pawn-only tree. A second is enough to see a jump, then the search stops.
        if my_walls + opp_walls == 0:
            return min(1.0, max(0.02, remaining * 0.02))
        # They cannot wall, we still hold some, and the static race is safely won.
        if opp_walls == 0 and my_d <= opp_d and not pawns_interact:
            return min(0.3, max(0.02, remaining * 0.005))

        # Our own moves still to come: the walk plus the walls likely to be traded.
        moves_left = max(8, my_d + (my_walls + opp_walls) // 2 + 4)
        sharp = abs(my_d - opp_d) <= 3 and my_walls + opp_walls >= 2
        budget = remaining / moves_left * (1.5 if sharp else 1.0)
        # Long thinks pay only while we still hold the walls that a deep line can use.
        total = self._clock_total
        if sharp:
            cap = total * (CAP_SHARP if my_walls >= 4 else CAP_SHARP_FEW_WALLS)
        else:
            cap = total * CAP_ORDINARY
        if opening:
            cap = min(cap, total * CAP_OPENING)
        # Cumulative plan: past schedule, stop thinking long until the clock catches up.
        if step <= TIME_PLAN_PLY:
            share = TIME_PLAN_SHARE * step / TIME_PLAN_PLY
        else:
            share = min(0.9, TIME_PLAN_SHARE + 0.01 * (step - TIME_PLAN_PLY))
        if total - remaining > total * (share + 0.01):
            cap = min(cap, total * CAP_OVER_PLAN)
        # The meeting of the pawns is where the time is needed; keep a reserve for it.
        if contact and remaining < total * CONTACT_RESERVE:
            cap = min(cap, remaining / 40)
        if remaining < 90:
            cap = min(cap, 4.0)
        if remaining < 30:
            cap = min(cap, 1.2)
        if remaining < 12:
            cap = min(cap, 0.4)
        budget = min(budget, cap)
        reserve = 20.0 if remaining > 60 else remaining * 0.25
        budget = min(budget, max(0.03, remaining - reserve))
        return max(0.03, budget)

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
