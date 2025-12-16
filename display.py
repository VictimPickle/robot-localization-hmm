import numpy as np
import matplotlib.pyplot as plt
from main import GridWorldHMM


def belief_to_position_grid(hmm, belief):
    """Convert belief distribution to spatial grid (sum probabilities by position)."""
    grid = np.zeros((hmm.H, hmm.W), dtype=float)
    for s_idx, p in enumerate(belief):
        x, y, d = hmm.state_to_tuple(s_idx)
        grid[y, x] += p
    return grid


def dir_symbol(orient_char):
    """Convert orientation character to unicode arrow."""
    mapping = {
        "N": "↑",
        "E": "→",
        "S": "↓",
        "W": "←",
    }
    return mapping.get(orient_char, "●")


def main():
    # Initialize HMM
    hmm = GridWorldHMM(
        width=4,
        height=5,
        seed=0,
        p_forward=0.7,
        p_side=0.25,
        p_forward_stay=0.05,
    )

    # Simulate trajectory
    T_steps = 200
    states, actions, observations = hmm.simulate_trajectory(T_steps)
    bel_hist = hmm.forward_filter(actions, observations)
    max_t = min(len(bel_hist), len(states))

    # Create figure
    fig, (ax_prob, ax_robot) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Belief (Left) / True Robot Position (Right)")

    state = {
        "t": 0,
        "hmm": hmm,
        "states": states,
        "bel_hist": bel_hist,
        "max_t": max_t,
        "ax_prob": ax_prob,
        "ax_robot": ax_robot,
    }

    def update_display():
        """Update both plots with current belief and true position."""
        t = state["t"]
        hmm = state["hmm"]
        states = state["states"]
        bel_hist = state["bel_hist"]
        ax_prob = state["ax_prob"]
        ax_robot = state["ax_robot"]

        # Belief plot
        ax_prob.clear()
        belief = bel_hist[t]
        prob_grid = belief_to_position_grid(hmm, belief)
        im = ax_prob.imshow(prob_grid, origin="upper", cmap="viridis")
        ax_prob.set_title(f"Belief at t = {t}")
        ax_prob.set_xticks(range(hmm.W))
        ax_prob.set_yticks(range(hmm.H))
        ax_prob.set_xlabel("x")
        ax_prob.set_ylabel("y")

        # Add text annotations
        max_val = prob_grid.max() if prob_grid.size > 0 else 1.0
        for y in range(hmm.H):
            for x in range(hmm.W):
                val = prob_grid[y, x]
                color = "white" if val > max_val / 2 else "black"
                ax_prob.text(
                    x, y,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8, color=color,
                )

        # True position plot
        ax_robot.clear()
        robot_grid = np.zeros((hmm.H, hmm.W), dtype=float)
        s_idx = states[t]
        x, y, d = hmm.state_to_tuple(s_idx)
        robot_grid[y, x] = 1.0
        ax_robot.imshow(robot_grid, origin="upper", cmap="viridis")
        ax_robot.set_title(f"True Robot Position at t = {t}")
        ax_robot.set_xticks(range(hmm.W))
        ax_robot.set_yticks(range(hmm.H))
        ax_robot.set_xlabel("x")
        ax_robot.set_ylabel("y")

        # Add direction symbol
        orient_char = hmm.ORIENTS[d]
        ax_robot.text(
            x, y,
            dir_symbol(orient_char),
            ha="center", va="center",
            fontsize=20, color="red",
        )

        fig.canvas.draw_idle()

    def on_key(event):
        """Handle keyboard input to advance through time."""
        if event.key == " " or event.key == "space":
            for i in range(5):
                state["t"] += 1
                if state["t"] >= state["max_t"]:
                    state["t"] = 0
            update_display()

    # Connect keyboard event
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_display()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
