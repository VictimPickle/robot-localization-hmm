import numpy as np

class GridWorldHMM:
    """
    Grid world HMM with:
    - 2D position (x, y)
    - 4 orientations: N, E, S, W
    - Control-dependent transitions (Forward, TurnLeft, TurnRight, Stay)
    - Two sensors: beacon (binary) + wall distance (0,1,2)
    """

    ORIENTS = ['N', 'E', 'S', 'W']

    def __init__(self,
                 width=4,
                 height=5,
                 beacon_positions=None,
                 p_forward=0.7,
                 p_side=0.25,
                 p_forward_stay=0.05,
                 p_turn_success=0.9,
                 beacon_probs=(0.85, 0.4, 0.1),
                 wall_sensor_table=None,
                 seed=None):
        self.W = width
        self.H = height
        self.n_orients = len(self.ORIENTS)
        self.n_states = self.W * self.H * self.n_orients

        if beacon_positions is None:
            # default: three beacons
            self.beacons = [(0, 0), (3, 2), (1, 4)]  # (x,y)
        else:
            self.beacons = beacon_positions

        self.p_forward = p_forward
        self.p_side = p_side
        self.p_forward_stay = p_forward_stay
        self.p_turn_success = p_turn_success

        # beacon_probs = (P(see|dist=0), P(see|dist=1), P(see|dist>=2))
        self.beacon_probs = beacon_probs

        if wall_sensor_table is None:
            # rows correspond to true distance class: 0,1,2,far (>=3)
            # columns correspond to observed distance: 0,1,2
            self.wall_sensor_table = np.array([
                [0.7, 0.2, 0.1],    # true 0
                [0.2, 0.6, 0.2],    # true 1
                [0.1, 0.3, 0.6],    # true 2
                [0.05, 0.15, 0.8],  # true >=3
            ], dtype=float)
        else:
            self.wall_sensor_table = np.asarray(wall_sensor_table, dtype=float)

        self.rng = np.random.default_rng(seed)

        # Precompute transition matrices for each action
        self.transition_matrices = {}
        for action in ['F', 'L', 'R', 'S']:
            self.transition_matrices[action] = self._build_transition_matrix_for_action(action)

    # ---------- utility: state index <-> (x,y,orient) ----------

    def state_to_tuple(self, idx):
        """Convert state index -> (x, y, orient_index)."""
        xy_states = self.W * self.H
        d = idx // xy_states
        rem = idx % xy_states
        y = rem // self.W
        x = rem % self.W
        return x, y, d

    def tuple_to_state(self, x, y, d):
        """Convert (x, y, orient_index) -> state index."""
        return d * (self.W * self.H) + y * self.W + x

    # ---------- geometry helpers ----------

    def _move_forward(self, x, y, d):
        """Return (nx, ny) for moving one step forward in orientation d; clamp to stay if outside grid."""
        dx, dy = 0, 0
        orient = self.ORIENTS[d]
        if orient == 'N':
            dy = -1
        elif orient == 'S':
            dy = 1
        elif orient == 'E':
            dx = 1
        elif orient == 'W':
            dx = -1

        nx, ny = x + dx, y + dy
        if 0 <= nx < self.W and 0 <= ny < self.H:
            return nx, ny
        return x, y  # hit wall -> stay

    def _left_dir(self, d):
        return (d - 1) % self.n_orients

    def _right_dir(self, d):
        return (d + 1) % self.n_orients

    def _manhattan_to_nearest_beacon(self, x, y):
        dists = [abs(x - bx) + abs(y - by) for (bx, by) in self.beacons]
        return min(dists)

    def _distance_to_wall_ahead(self, x, y, d):
        """True integer distance from (x,y) to nearest wall in front, in cells."""
        dx, dy = 0, 0
        orient = self.ORIENTS[d]
        if orient == 'N':
            dy = -1
        elif orient == 'S':
            dy = 1
        elif orient == 'E':
            dx = 1
        elif orient == 'W':
            dx = -1

        dist = 0
        cx, cy = x, y
        while True:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < self.W and 0 <= ny < self.H):
                return dist
            dist += 1
            cx, cy = nx, ny

    # ---------- transition model ----------

    def _build_transition_matrix_for_action(self, action):
        """
        Build an (n_states x n_states) matrix T where:
        T[i, j] = P(S_t = j | S_{t-1} = i, action_t = action)
        Rows: from-state, Cols: to-state
        """
        T = np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            x, y, d = self.state_to_tuple(i)
            if action == 'F':
                # Forward
                nx, ny = self._move_forward(x, y, d)
                j = self.tuple_to_state(nx, ny, d)
                T[i, j] += self.p_forward
                # Left forward
                d_left = self._left_dir(d)
                nx_left, ny_left = self._move_forward(x, y, d_left)
                j_left = self.tuple_to_state(nx_left, ny_left, d_left)
                T[i, j_left] += self.p_side / 2.0
                # Right forward
                d_right = self._right_dir(d)
                nx_right, ny_right = self._move_forward(x, y, d_right)
                j_right = self.tuple_to_state(nx_right, ny_right, d_right)
                T[i, j_right] += self.p_side / 2.0
                # Stay
                j_stay = self.tuple_to_state(x, y, d)
                T[i, j_stay] += self.p_forward_stay
            elif action == 'L':
                # Turn left
                d_new = self._left_dir(d)
                j_success = self.tuple_to_state(x, y, d_new)
                T[i, j_success] += self.p_turn_success
                j_fail = self.tuple_to_state(x, y, d)
                T[i, j_fail] += (1.0 - self.p_turn_success)
            elif action == 'R':
                # Turn right
                d_new = self._right_dir(d)
                j_success = self.tuple_to_state(x, y, d_new)
                T[i, j_success] += self.p_turn_success
                j_fail = self.tuple_to_state(x, y, d)
                T[i, j_fail] += (1.0 - self.p_turn_success)
            elif action == 'S':
                # Stay
                j = self.tuple_to_state(x, y, d)
                T[i, j] = 1.0

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = T / row_sums

        return T

    # ---------- observation model ----------

    def observation_prob(self, state_idx, obs):
        """
        P(obs | state).
        obs = (see_beacon: bool, wall_obs: int in {0,1,2})
        """
        x, y, d = self.state_to_tuple(state_idx)
        see_beacon, wall_obs = obs

        # Beacon probability
        dist_b = self._manhattan_to_nearest_beacon(x, y)
        if dist_b == 0:
            p_see = self.beacon_probs[0]
        elif dist_b == 1:
            p_see = self.beacon_probs[1]
        else:
            p_see = self.beacon_probs[2]

        if see_beacon:
            p_beacon_obs = p_see
        else:
            p_beacon_obs = 1.0 - p_see

        # Wall distance probability
        true_dist = self._distance_to_wall_ahead(x, y, d)
        if true_dist == 0:
            row = self.wall_sensor_table[0]
        elif true_dist == 1:
            row = self.wall_sensor_table[1]
        elif true_dist == 2:
            row = self.wall_sensor_table[2]
        else:
            row = self.wall_sensor_table[3]

        p_wall_obs = row[wall_obs]

        return p_beacon_obs * p_wall_obs

    # ---------- simulation ----------

    def simulate_step(self, state_idx, action):
        """Sample next state given current state and action."""
        T = self.transition_matrices[action]
        probs = T[state_idx]
        next_state = self.rng.choice(self.n_states, p=probs)
        return next_state

    def simulate_observation(self, state_idx):
        """Sample observation (see_beacon, wall_obs) given state."""
        x, y, d = self.state_to_tuple(state_idx)

        # Beacon
        dist_b = self._manhattan_to_nearest_beacon(x, y)
        if dist_b == 0:
            p_see = self.beacon_probs[0]
        elif dist_b == 1:
            p_see = self.beacon_probs[1]
        else:
            p_see = self.beacon_probs[2]

        see_beacon = self.rng.random() < p_see

        # Wall
        true_dist = self._distance_to_wall_ahead(x, y, d)
        if true_dist == 0:
            row = self.wall_sensor_table[0]
        elif true_dist == 1:
            row = self.wall_sensor_table[1]
        elif true_dist == 2:
            row = self.wall_sensor_table[2]
        else:
            row = self.wall_sensor_table[3]

        wall_obs = self.rng.choice([0, 1, 2], p=row)
        return (see_beacon, int(wall_obs))

    def simulate_trajectory(self, T_steps, start_state=None, actions=None):
        """
        Simulate a trajectory of length T_steps:
        returns states[0..T], actions[1..T], observations[1..T]
        """
        if start_state is None:
            # uniform random start
            start_state = self.rng.integers(0, self.n_states)

        if actions is None:
            # random actions of {F, L, R, S}
            actions = self.rng.choice(['F', 'L', 'R', 'S'],
                                      size=T_steps,
                                      replace=True).tolist()
        else:
            assert len(actions) == T_steps

        states = [start_state]
        observations = []
        cur = start_state

        for t in range(T_steps):
            a = actions[t]
            cur = self.simulate_step(cur, a)
            states.append(cur)
            obs = self.simulate_observation(cur)
            observations.append(obs)

        return states, actions, observations

    # ---------- inference: forward (filtering) ----------

    def forward_filter(self, actions, observations, prior=None):
        """
        Compute belief over states for each time t using forward algorithm.
        actions: list of actions a_1..a_T
        observations: list of obs O_1..O_T
        prior: np.array of shape (n_states,), sums to 1
        returns: beliefs: list of np.array, length T+1 (includes Bel(S_0))
        """
        T_steps = len(actions)
        assert len(observations) == T_steps

        if prior is None:
            prior = np.ones(self.n_states, dtype=float) / self.n_states
        else:
            prior = np.asarray(prior, dtype=float)
            prior = prior / prior.sum()

        beliefs = [prior.copy()]
        bel = prior.copy()

        for t in range(T_steps):
            a_t = actions[t]
            o_t = observations[t]

            T_a = self.transition_matrices[a_t]
            bel_pred = T_a.T @ bel

            bel_new = np.zeros(self.n_states, dtype=float)
            for s in range(self.n_states):
                p_obs = self.observation_prob(s, o_t)
                bel_new[s] = p_obs * bel_pred[s]

            total = bel_new.sum()
            if total > 0:
                bel_new = bel_new / total
            else:
                bel_new = np.ones(self.n_states, dtype=float) / self.n_states

            bel = bel_new
            beliefs.append(bel.copy())

        return beliefs

    # ---------- inference: Viterbi (most likely path) ----------

    def viterbi(self, actions, observations, prior=None):
        """
        Compute most likely state sequence using Viterbi.
        Returns:
        path_states: list of state indices S_0..S_T
        log_prob: log probability of best path
        """
        T_steps = len(actions)
        assert len(observations) == T_steps

        if prior is None:
            prior = np.ones(self.n_states, dtype=float) / self.n_states
        else:
            prior = np.asarray(prior, dtype=float)
            prior = prior / prior.sum()

        # precompute log transitions
        log_T = {a: np.log(self.transition_matrices[a] + 1e-12)
                 for a in self.transition_matrices.keys()}

        # delta[t, s] = best log prob of path ending in state s at time t
        # we index time t from 0..T, with t=0 the initial state before any obs
        delta = np.full((T_steps + 1, self.n_states),
                        -np.inf, dtype=float)
        psi = np.full((T_steps + 1, self.n_states),
                      -1, dtype=int)

        # initialization with prior and no observation at t=0
        delta[0] = np.log(prior + 1e-12)
        psi[0] = -1

        # recursion
        for t in range(1, T_steps + 1):
            a_t = actions[t - 1]
            o_t = observations[t - 1]
            log_T_a = log_T[a_t]

            # precompute log obs likelihoods for time t
            log_obs = np.zeros(self.n_states, dtype=float)
            for s in range(self.n_states):
                p = self.observation_prob(s, o_t)
                log_obs[s] = np.log(p + 1e-12)

            for s_to in range(self.n_states):
                # best predecessor
                scores = delta[t - 1] + log_T_a[:, s_to]
                best_prev = np.argmax(scores)
                delta[t, s_to] = scores[best_prev] + log_obs[s_to]
                psi[t, s_to] = best_prev

        # termination
        last_t = T_steps
        best_last_state = np.argmax(delta[last_t])
        best_log_prob = delta[last_t, best_last_state]

        # backtrack
        path = [best_last_state]
        cur = best_last_state
        for t in range(last_t, 0, -1):
            cur = psi[t, cur]
            path.append(cur)
        path.reverse()  # from S_0..S_T

        return path, best_log_prob

    # ---------- demo ----------

    def demo(self):
        """Run a demo of the HMM filtering and Viterbi decoding."""
        T_steps = 1000
        actions = None
        states, actions, observations = self.simulate_trajectory(T_steps, actions=actions)

        print("=== Simulated trajectory ===")
        for t in range(min(20, T_steps + 1)):
            x, y, d = self.state_to_tuple(states[t])
            if t == 0:
                print(f"t={t:2d} STATE (true) = (x={x}, y={y}, dir={self.ORIENTS[d]}) (no obs)")
            else:
                a = actions[t - 1]
                o = observations[t - 1]
                print(f"t={t:2d} a={a} STATE (true) = (x={x}, y={y}, dir={self.ORIENTS[d]}), obs={o}")

        # Forward filtering
        beliefs = self.forward_filter(actions, observations)
        print("\n=== Filtering: argmax state at each t ===")
        for t in range(min(20, len(beliefs))):
            bel = beliefs[t]
            s_hat = int(np.argmax(bel))
            x, y, d = self.state_to_tuple(s_hat)
            print(f"t={t:2d} MAP state = (x={x}, y={y}, dir={self.ORIENTS[d]}), prob={bel[s_hat]:.3f}")

        # Viterbi decoding
        v_path, logp = self.viterbi(actions, observations)
        print("\n=== Viterbi most likely path (first 20 steps) ===")
        print(f"log P(path*, obs | model) = {logp:.3f}")
        for t in range(min(20, len(v_path))):
            s = v_path[t]
            x, y, d = self.state_to_tuple(s)
            print(f"t={t:2d} Viterbi state = (x={x}, y={y}, dir={self.ORIENTS[d]})")

        # Compare Viterbi with true states
        print("\n=== Compare Viterbi path with true states ===")
        correct = 0
        for t in range(T_steps + 1):
            s_true = states[t]
            s_v = v_path[t]
            match = (s_true == s_v)
            if match:
                correct += 1
            if t < 20:
                xt, yt, dt = self.state_to_tuple(s_true)
                xv, yv, dv = self.state_to_tuple(s_v)
                print(f"t={t:2d} TRUE=(x={xt},y={yt},{self.ORIENTS[dt]}) "
                      f"VIT=(x={xv},y={yv},{self.ORIENTS[dv]}) match={match}")
        print(f"\nTotal matches: {correct}/{T_steps+1}")


if __name__ == "__main__":
    hmm = GridWorldHMM(seed=42)
    hmm.demo()
