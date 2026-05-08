"""GridWorld environment for Coverage Path Planning (CPP).

Modifications relative to the original `grid_world_cpp.py`:

1. The observation space is a `Dict` with three components whose shapes do
   NOT depend on the grid size. This is the key change that makes a single
   policy transferable between 5x5, 10x10 and 20x20 grids:

       - "agent_pos":  (2,)            float, normalized (x/size, y/size)
       - "coverage":   (1,)            float, fraction of free cells visited
       - "local_view": (k, k)          int8, agent-centered patch of size
                                       `local_view_size` (default 5)

   The local view encodes:
       0 = free, not visited
       1 = obstacle or out-of-grid (wall)
       2 = visited (already covered)
       3 = current agent cell  (always at the patch center)

   With k = 5, the patch is identical in shape regardless of grid size, so
   the same CNN weights apply to every stage of the curriculum.

2. The agent still has a strictly partial view of the world, satisfying the
   assignment's constraint. We only widen the patch from 3x3 to 5x5; we do
   NOT give it the full map.

3. The reward function is unchanged from the project description (new cell
   +1.0, revisit -0.3, collision -0.5, step -0.1, +10.0 coverage complete,
   timeout -5.0), so behavior is comparable to the baseline.

The env is registered as a regular Gymnasium env via the class import in
the gymnasium_env package; no extra registration is required by the
training script (it instantiates `GridWorldCPPEnv` directly).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame  # only needed for render_mode="human"
except Exception:  # pragma: no cover - pygame is optional
    pygame = None


# Cell encodings used inside `local_view` (and internally).
FREE = 0
WALL = 1   # obstacle OR out-of-grid
VISITED = 2
AGENT = 3


class GridWorldCPPEnv(gym.Env):
    """Coverage Path Planning on a square grid with random obstacles.

    Parameters
    ----------
    size : int
        Grid side length (5, 10, 20, ...).
    obs_quantity : int
        Number of obstacle cells placed uniformly at random at reset.
    max_steps : int
        Episode step cap; on cap, returns truncated=True with -5 reward.
    local_view_size : int
        Side length of the agent-centered patch returned in the observation.
        Must be odd (so the agent sits at the center). Default 5.
    render_mode : Optional[str]
        Either None or "human" (uses pygame).
    """

    metadata = {"render_modes": ["human"], "render_fps": 8}

    # action -> (dx, dy). Matches the original env so train_grid_world_cpp.py's
    # `print_action` mapping (right/up/left/down) stays meaningful.
    _ACTION_TO_DELTA = {
        0: np.array([+1,  0]),  # right
        1: np.array([ 0, -1]),  # up
        2: np.array([-1,  0]),  # left
        3: np.array([ 0, +1]),  # down
    }

    def __init__(
        self,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
        local_view_size: int = 5,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if local_view_size % 2 == 0 or local_view_size < 3:
            raise ValueError("local_view_size must be an odd integer >= 3")

        self.size = int(size)
        self.obs_quantity = int(obs_quantity)
        self.max_steps = int(max_steps)
        self.local_view_size = int(local_view_size)
        self._half_view = self.local_view_size // 2

        self.action_space = spaces.Discrete(4)

        # Size-INVARIANT observation. Crucial for transfer learning between
        # grids: the policy network sees the same input shape on every stage.
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "coverage": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "local_view": spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.local_view_size, self.local_view_size),
                    dtype=np.int8,
                ),
            }
        )

        self.render_mode = render_mode
        self._window = None
        self._clock = None
        self._cell_pixels = 48

        # Filled by reset()
        self._grid: Optional[np.ndarray] = None      # WALL where obstacle
        self._visited: Optional[np.ndarray] = None   # bool
        self._agent: Optional[np.ndarray] = None     # (x, y) ints
        self._steps: int = 0
        self._total_free: int = 0

    # ------------------------------------------------------------------ utils

    def _free_cells_count(self) -> int:
        return int((self._grid == FREE).sum())

    def _visited_count(self) -> int:
        # Only count free cells that have been visited; an obstacle can never
        # be visited so this is just self._visited.sum() in practice.
        return int(self._visited.sum())

    def _coverage_ratio(self) -> float:
        return self._visited_count() / max(1, self._total_free)

    def _build_local_view(self) -> np.ndarray:
        """Return an (k, k) int8 patch centered on the agent.

        Out-of-grid cells are encoded as WALL so the agent can learn map
        boundaries from the patch alone.
        """
        k = self.local_view_size
        h = self._half_view
        ax, ay = int(self._agent[0]), int(self._agent[1])

        view = np.full((k, k), WALL, dtype=np.int8)
        for dy in range(-h, h + 1):
            for dx in range(-h, h + 1):
                gx, gy = ax + dx, ay + dy
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    if self._grid[gx, gy] == WALL:
                        view[dy + h, dx + h] = WALL
                    elif self._visited[gx, gy]:
                        view[dy + h, dx + h] = VISITED
                    else:
                        view[dy + h, dx + h] = FREE
        # Agent always sits at the patch center.
        view[h, h] = AGENT
        return view

    def _get_obs(self) -> dict:
        return {
            "agent_pos": np.array(
                [self._agent[0] / self.size, self._agent[1] / self.size],
                dtype=np.float32,
            ),
            "coverage": np.array([self._coverage_ratio()], dtype=np.float32),
            "local_view": self._build_local_view(),
        }

    def _get_info(self) -> dict:
        return {
            "coverage": self._coverage_ratio(),
            "visited": self._visited_count(),
            "total_free": self._total_free,
            "steps": self._steps,
        }

    # ----------------------------------------------------------------- gym API

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        # Build empty grid, drop obstacles uniformly at random.
        self._grid = np.full((self.size, self.size), FREE, dtype=np.int8)
        all_cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.np_random.shuffle(all_cells)

        # Reserve a random spawn cell for the agent before placing obstacles
        # so we never spawn on top of a wall.
        spawn_idx = 0
        spawn = all_cells[spawn_idx]
        self._agent = np.array(spawn, dtype=np.int64)

        # Place obstacles on the next obs_quantity cells (skipping the spawn).
        placed = 0
        for cell in all_cells[1:]:
            if placed >= self.obs_quantity:
                break
            self._grid[cell[0], cell[1]] = WALL
            placed += 1

        self._visited = np.zeros((self.size, self.size), dtype=bool)
        self._visited[spawn[0], spawn[1]] = True  # agent's starting cell counts

        self._total_free = self._free_cells_count()
        self._steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        if self._grid is None:
            raise RuntimeError("step() called before reset()")

        action = int(action)
        delta = self._ACTION_TO_DELTA[action]
        proposed = self._agent + delta

        # Per-step penalty (-0.1) is applied unconditionally.
        reward = -0.1
        terminated = False
        truncated = False

        # Out of grid OR obstacle -> collision: stay put, -0.5 extra.
        if (
            proposed[0] < 0
            or proposed[0] >= self.size
            or proposed[1] < 0
            or proposed[1] >= self.size
            or self._grid[proposed[0], proposed[1]] == WALL
        ):
            reward += -0.5
        else:
            self._agent = proposed
            ax, ay = int(self._agent[0]), int(self._agent[1])
            if not self._visited[ax, ay]:
                # New cell: +1.0
                self._visited[ax, ay] = True
                reward += 1.0
            else:
                # Revisit: -0.3
                reward += -0.3

        self._steps += 1

        # Termination: full coverage of free cells.
        if self._visited_count() >= self._total_free:
            reward += 10.0
            terminated = True

        # Truncation: ran out of steps without full coverage.
        if not terminated and self._steps >= self.max_steps:
            reward += -5.0
            truncated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # -------------------------------------------------------------- rendering

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        return None

    def _render_frame(self):
        if pygame is None:
            return None
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self.size * self._cell_pixels,
                 self.size * self._cell_pixels + 30)
            )
            pygame.display.set_caption("GridWorld CPP")
        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (self.size * self._cell_pixels,
             self.size * self._cell_pixels + 30)
        )
        canvas.fill((255, 255, 255))

        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(
                    x * self._cell_pixels,
                    y * self._cell_pixels + 30,
                    self._cell_pixels, self._cell_pixels,
                )
                if self._grid[x, y] == WALL:
                    pygame.draw.rect(canvas, (0, 0, 0), rect)
                elif self._visited[x, y]:
                    pygame.draw.rect(canvas, (170, 230, 170), rect)
                else:
                    pygame.draw.rect(canvas, (255, 255, 255), rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        ax, ay = int(self._agent[0]), int(self._agent[1])
        center = (
            ax * self._cell_pixels + self._cell_pixels // 2,
            ay * self._cell_pixels + self._cell_pixels // 2 + 30,
        )
        pygame.draw.circle(canvas, (50, 110, 220), center, self._cell_pixels // 3)

        font = pygame.font.SysFont(None, 22)
        info = font.render(
            f"coverage: {self._coverage_ratio():.1%}   steps: {self._steps}",
            True, (0, 0, 0),
        )
        canvas.blit(info, (8, 6))

        self._window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._window is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
