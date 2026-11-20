"""Quoridor agent: iterative-deepening principal-variation search on a bitboard.

Cells are bits of one integer (bit ``r * 9 + c``). Two masks hold the open
edges: ``vmask`` bit ``i`` means ``i <-> i + 9`` is open, ``hmask`` bit ``i``
means ``i <-> i + 1`` is open. A breadth-first layer is then four shifts and
masks, so path lengths cost a handful of integer operations per step.
"""

from __future__ import annotations

import json
import os
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
# Worth of the walls still in hand, indexed by count. A wall that adds one step
# to their path trades evenly with one pawn step of ours, so what it really
# costs is the wall itself: the marginal value below. Late walls are worth far
# more, because only then does the opponent's path have a short route next to
# a much longer one, and a single wall can switch between them.
# Marginals, tenth wall to last: 60 80 100 130 170 220 270 330 380 420.
WALL_VALUE = (0, 420, 800, 1130, 1400, 1620, 1790, 1920, 2020, 2100, 2160)
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
CAP_SHARP = 60 / 900
CAP_SHARP_FEW_WALLS = 24 / 900
CAP_ORDINARY = 12 / 900
CAP_OPENING = 6 / 900
CAP_OVER_PLAN = 4 / 900
# Kept back until the pawns have passed each other.
CONTACT_RESERVE = 150 / 900
# Per step beyond the first that the best single wall would add to a path.
# STRIKE when the wall's owner moves next, DODGE when the victim does.
THREAT_STRIKE = 60
THREAT_DODGE = 25
THREAT_CAP = 7
TT_LIMIT = 1_200_000
# One JSON line per move in logs/ next to this file.
MOVE_LOG = True

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
    import random

    rng = random.Random(0xC0FFEE)
    def rnd():
        return rng.getrandbits(64)

    pawns = [[rnd() for _ in range(81)] for _ in range(2)]
    horizontal = [rnd() for _ in range(64)]
    vertical = [rnd() for _ in range(64)]
    walls_left = [[rnd() for _ in range(11)] for _ in range(2)]
    side = rnd()
    return pawns, horizontal, vertical, walls_left, side


Z_PAWN, Z_H, Z_V, Z_LEFT, Z_SIDE = _zobrist_tables()


def _edge_walls():
    """Walls that close each directed step, matching ``_candidate_blocking_walls``."""
    table = {}
    for r in range(9):
        for c in range(9):
            a = r * 9 + c
            if r < 8:
                walls = []
                if c > 0:
                    walls.append(H_BASE + r * 8 + (c - 1))
                if c < 8:
                    walls.append(H_BASE + r * 8 + c)
                table[(a, a + 9)] = tuple(walls)
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
                table[(a, a + 1)] = tuple(walls)
            if c > 0:
                walls = []
                if r > 0:
                    walls.append(V_BASE + (r - 1) * 8 + (c - 1))
                if r < 8:
                    walls.append(V_BASE + r * 8 + (c - 1))
                table[(a, a - 1)] = tuple(walls)
    return table


EDGE_WALLS = _edge_walls()


class SearchTimeout(Exception):
    """Raised when the move deadline is reached."""


class Engine:
    """Mutable board used only inside the search. ``load`` rebuilds it each turn."""

    __slots__ = (
        "vmask", "hmask", "h_walls", "v_walls", "pawns", "goals",
        "walls_left", "side", "ply", "zhash", "cache", "paths", "tt", "history",
        "killers", "deadline", "nodes", "ext", "dcache", "root_pawn",
        "completed_depth", "completed_score", "line", "played", "contempt", "banned",
    )

    def __init__(self, cache: dict, paths: dict, tt: dict, history: list[int]) -> None:
        self.cache = cache
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
        if len(dcache) > 600_000:
            dcache.clear()
        dcache[key] = d
        return d

    def _race(self, start: int, goal_row: int):
        """Pawn-ignoring (distance, narrowest layer of the shortest-path DAG).

        ``(None, 0)`` when the row is unreachable. A narrowest layer of one cell
        means every shortest path squeezes through that cell.
        """
        v, h = self.vmask, self.hmask
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
        if len(dcache) > 600_000:
            dcache.clear()
        dcache[key] = result
        return result

    def _path(self, start: int, goal_row: int):
        """One shortest path: length, narrowest layer, first step, walls that cut it.

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
        if len(paths) > 300_000:
            paths.clear()
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
        back = front & goal
        narrow = _popcount(back)
        cur = back & -back
        cells = [cur.bit_length() - 1]
        for k in range(len(layers) - 1, 0, -1):
            back = layers[k] & (((back >> 9) & v) | ((back & v) << 9)
                                | ((back >> 1) & h) | ((back & h) << 1))
            width = _popcount(back)
            if width < narrow:
                narrow = width
            step = layers[k] & (((cur >> 9) & v) | ((cur & v) << 9)
                                | ((cur >> 1) & h) | ((cur & h) << 1))
            cur = step & -step
            cells.append(cur.bit_length() - 1)
        cells.append(start)
        blockers = []
        seen_walls = set()
        for i in range(len(cells) - 1):
            for wall in EDGE_WALLS[(cells[i + 1], cells[i])]:
                if wall not in seen_walls:
                    seen_walls.add(wall)
                    blockers.append(wall)
        return (len(layers), narrow, cells[-2], tuple(blockers))

    def _threat(self, start: int, goal_row: int, d: int) -> int:
        """Most steps one more wall can add to this pawn's path."""
        v, h = self.vmask, self.hmask
        key = (start, goal_row + 32, v, h)
        dcache = self.dcache
        hit = dcache.get(key)
        if hit is not None:
            return hit
        worst = 0
        if d:
            # A wall that lengthens the path cuts every shortest path, this one included.
            for wall in self._path(start, goal_row)[3]:
                if not self.geometry_ok(wall):
                    continue
                nv, nh = self._masks_after(wall)
                nd = self._dist(start, goal_row, nv, nh)
                if nd is not None and nd - d > worst:
                    worst = nd - d
        if len(dcache) > 600_000:
            dcache.clear()
        dcache[key] = worst
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
        if len(dcache) > 600_000:
            dcache.clear()
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
        """True when a jump or a blocked landing can still change the static race.

        Distance ignores the other pawn, so an equal race is a proven win only
        after the pawns have passed, or the leader finishes before a jump can
        land, or neither pawn lies on the other's shortest path and the row
        gap is too wide for a one-step approach.
        """
        p0 = self.pawns[0]
        p1 = self.pawns[1]
        r0 = ROW[p0]
        r1 = ROW[p1]
        gap = abs(r0 - r1)
        if p0 != p1 and max(gap, abs((p0 % 9) - (p1 % 9))) <= 2:
            return True
        faces0 = (r1 - r0) * (self.goals[0] - r0) > 0
        faces1 = (r0 - r1) * (self.goals[1] - r1) > 0
        if not faces0 and not faces1:
            return False
        leader = d0 if d0 <= d1 else d1
        if leader < max(0, gap - 2):
            return False
        if self._on_shortest_path(p1, p0, self.goals[0], d0):
            return True
        if self._on_shortest_path(p0, p1, self.goals[1], d1):
            return True
        return gap <= 3

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

    def evaluate(self) -> int:
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
            return FORCED_RACE + 10_000 - my_d * 20
        if my_left == 0 and adv < 0 and not self._pawns_interact(d0, d1):
            return -(FORCED_RACE + 10_000) + opp_d * 20 - my_d

        score = adv * STEP - my_d * 3 + WALL_VALUE[my_left] - WALL_VALUE[opp_left]
        # A pawn in a corridor one wall can seal is far worse off than its
        # distance says: the holder of that wall plays it at the last moment.
        # A wall closes two edges; with three or more cells in every layer of the
        # shortest-path graph, each layer change keeps an open edge.
        # The side to move can still step out of a trap or drop its own wall first,
        # so its exposure counts less than the one it can spring right now.
        if opp_left and my_narrow <= 2:
            threat = self._threat(self.pawns[side], self.goals[side], my_d)
            if threat > 1:
                score -= (min(threat, THREAT_CAP) - 1) * THREAT_DODGE
            elif my_narrow <= 1:
                score -= 20
        if my_left and opp_narrow <= 2:
            threat = self._threat(self.pawns[opp], self.goals[opp], opp_d)
            if threat > 1:
                score += (min(threat, THREAT_CAP) - 1) * THREAT_STRIKE
            elif opp_narrow <= 1:
                score += 16
        score -= abs((self.pawns[side] % 9) - 4)
        return score

    def _collect_walls(self, depth: int) -> list[tuple[int, int]]:
        side = self.side
        if self.walls_left[side] <= 0:
            return []
        opp = side ^ 1
        info = self.analyze()
        my_d = info[side][0]
        opp_d = info[opp][0]
        opp_narrow = info[opp][1]
        blockers = info[opp][3]
        if my_d is None or opp_d is None or not blockers:
            return []
        # Ahead, and they have nothing left to drop in front of us: run.
        if self.walls_left[opp] == 0 and opp_d >= my_d:
            return []

        root = self.ply == 0
        critical = opp_d <= 2
        # Deep in the tree, the walls that matter are the ones in front of the
        # pawn and the ones sealing the goal. The root sees every candidate.
        if not root and not critical and depth < 6 and len(blockers) > 8:
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
        my_blocks = set(info[side][3])
        pawns = self.pawns
        goals = self.goals
        dist = self._dist
        scored: list[tuple[int, int]] = []
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
            if critical and delta_opp >= 1:
                priority = 1_800_000 + delta_opp * 10_000 - delta_my * 1_000
            elif delta_opp >= 1 and gain >= 1:
                priority = 1_500_000 + gain * 10_000 + delta_opp * 100 - delta_my * 100
            elif (
                (depth >= 6 or root)
                and delta_my == 0
                and delta_opp == 0
                and opp_narrow >= 2
            ):
                # The wall sits on their current route but a twin route remains.
                # Keep a couple of these so a follow-up wall is visible a few plies later.
                priority = 700_000
            else:
                continue
            priority += min(self.history[wall], 4_000) >> 4
            scored.append((priority, wall))

        if not scored:
            return []
        scored.sort(reverse=True)
        if root:
            limit = 14
        elif critical:
            limit = 10
        elif depth >= 8:
            limit = 7
        elif depth >= 5:
            limit = 5
        elif depth >= 3:
            limit = 4
        else:
            limit = 3
        return scored[:limit]

    def generate(self, depth: int, tt_move: int) -> list[int]:
        side = self.side
        goal = self.goals[side]
        info = self.analyze()
        my_step = info[side][2]
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
            if killer < 0 or killer == tt_move or not self._plausible(killer):
                continue
            if killer in ordered:
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
                return self.evaluate()

            best = -INF
            best_move = -1
            node_ply = self.ply
            # Hash move first. A cutoff here skips wall generation entirely.
            if tt_move >= 0 and self._plausible(tt_move):
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
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    self._push_killer(node_ply, tt_move)
                    self.history[tt_move] += depth * depth

            if alpha < beta:
                moves = self.generate(depth, tt_move)
                first = best_move < 0
                for move in moves:
                    if move == best_move:
                        continue
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

    def _root_bias(self, move: int, my_d: int) -> int:
        """Shuffling away from the goal hands the opponent free tempo; demand a reason."""
        if move >= 81 or my_d is None:
            return 0
        after = self._dist(move, self.goals[0], self.vmask, self.hmask)
        if after is not None and after > my_d:
            return -REGRESS_PENALTY
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

    def search_root(self, depth: int):
        entry = self.tt.get(self.zhash)
        tt_move = entry[3] if entry is not None else -1
        moves = self.root_moves(depth, tt_move)
        if not moves:
            raise RuntimeError("No legal move.")
        my_d = self.distances()[0]

        alpha = -INF
        best_move = None
        best_score = -INF
        for index, move in enumerate(moves):
            if time.perf_counter() >= self.deadline and best_move is not None:
                break
            bias = self._root_bias(move, my_d)
            # Search for the raw score that would beat alpha once the bias is added.
            target = alpha - bias
            token = self.apply(move)
            try:
                if token[1]:
                    score = MATE - self.ply
                elif index == 0:
                    score = -self.negamax(depth - 1, -INF, INF)
                else:
                    score = -self.negamax(depth - 1, -target - 1, -target)
                    if score > target:
                        score = -self.negamax(depth - 1, -INF, -target)
            except SearchTimeout:
                self.undo(token)
                if best_move is None:
                    raise
                break
            else:
                self.undo(token)
            if -MATE_BOUND < score < MATE_BOUND:
                score += bias
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
        return best_move, best_score

    def choose(self, budget: float):
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
        # Best move of the deepest iteration that still saw a game left to play.
        hopeful = None
        hopeful_depth = 0
        info = self.analyze()
        step = info[0][2]
        if step >= 0 and not self._is_banned(step):
            best_move = step

        for depth in range(1, 32):
            now = time.perf_counter()
            # The next iteration costs several times the last one; do not start
            # what cannot finish.
            if now >= self.deadline or (depth > 2 and now - start > 0.45 * budget):
                break
            try:
                move, score = self.search_root(depth)
            except SearchTimeout:
                break
            if move is None:
                break
            best_move = move
            best_score = score
            self.completed_depth = depth
            self.completed_score = score
            if score > -3 * STEP:
                hopeful = move
                hopeful_depth = depth
            if score >= MATE - 40 or score >= FORCED_RACE:
                break
            if score <= -(MATE - 40):
                break
        # Against perfect play every move loses. Keep the line that looked
        # close a ply shorter, so a single slip still lets us through. A
        # shallower or already-bleak hope is noise: keep the race as close as possible.
        if best_score <= -FORCED_RACE:
            if hopeful is not None and hopeful_depth >= self.completed_depth - 1:
                best_move = hopeful
            else:
                best_move = self._delaying_move(best_move)
        if best_move is None:
            dests = self._pawn_destinations()
            free = [dest for dest in dests if not self._is_banned(dest)]
            best_move = (free or dests or [0])[0]
        return best_move

    def _delaying_move(self, fallback):
        """Every line loses: keep the race as close as the board allows, so a slip can still be punished."""
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
                r0, r1 = self.races()
            finally:
                self.undo(token)
            mine, theirs = r0[0], r1[0]
            if mine is None or theirs is None:
                continue
            walls = self.walls_left[0] - (move >= 81)
            key = ((theirs - mine) * STEP + WALL_VALUE[walls], move < 81)
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
        self._cache: dict = {}
        self._paths: dict = {}
        self._tt: dict = {}
        self._history = [0] * 264
        self._engine = Engine(self._cache, self._paths, self._tt, self._history)
        self.last_depth = 0
        self.last_score = 0
        self.last_nodes = 0
        self.last_move = ""
        # Visits of every position this game, both sides to move. Kept across
        # walls: the hash holds the walls, so old entries simply never match.
        self._visits: dict[int, int] = {}
        self._last_step = -1
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
        if step < self._last_step or self._last_step < 0:
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

        info = engine.analyze()
        my_d = info[0][0] or 8
        opp_d = info[1][0] or 8
        my_walls, opp_walls = engine.walls_left
        opening = not current_state.rep.walls and min(my_d, opp_d) >= 5
        contact = engine._pawns_facing()
        budget = self._budget(remaining_time, my_d, opp_d, my_walls, opp_walls, opening, step, contact)
        if visits >= 2:
            # Searched before, and the table still holds it.
            budget = min(budget, REPEAT_BUDGET)

        try:
            move = engine.choose(budget)
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
                opening: bool = False, step: int = 0, contact: bool = False) -> float:
        if remaining <= 0:
            return 0.02
        # Pawn-only tree. A second is enough to see a jump, then the search stops.
        if my_walls + opp_walls == 0:
            return min(1.0, max(0.02, remaining * 0.02))
        # They cannot wall, we still hold some, and the static race is not behind.
        if opp_walls == 0 and my_d <= opp_d:
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
