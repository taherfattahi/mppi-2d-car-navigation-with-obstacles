import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# 1. Car dynamics
# -------------------------
def car_dynamics(state, steering_angle, dt=0.1, v=1.0):
    """
    Simple 2D car model with constant forward speed v.
    state = (x, y, theta)
    steering_angle = angular velocity (turn rate)
    """
    x, y, theta = state
    theta_new = theta + steering_angle * dt
    x_new = x + v * np.cos(theta_new) * dt
    y_new = y + v * np.sin(theta_new) * dt
    return np.array([x_new, y_new, theta_new])


# -------------------------
# 2. Cost function (multi-obstacles)
# -------------------------
def compute_cost(states, goal, obstacles=None, penalty=100.0):
    """
    Cost = final distance to goal
           + large penalty if colliding with ANY obstacle.

    obstacles: list of dicts or tuples with 'center' and 'radius', e.g.:
       obstacles = [
         {"center": np.array([ox, oy]), "radius": r},
         ...
       ]
       or you can store them however you prefer.
    """
    # Distance to goal (only final state, for simplicity)
    final_dist_goal = np.linalg.norm(states[-1, 0:2] - goal[0:2])
    cost = final_dist_goal
    
    # Collision check with multiple obstacles
    if obstacles is not None:
        for obs in obstacles:
            center = obs["center"]
            radius = obs["radius"]
            # Distances from each state to this obstacle
            dists = np.linalg.norm(states[:, 0:2] - center, axis=1)
            # If any state is within radius => collision
            if np.any(dists < radius):
                cost += penalty
                break  # No need to check other obstacles if we already collided

    return cost


# -------------------------
# 3. One MPPI iteration
# -------------------------
def mppi_iteration(
    init_state, 
    goal_state, 
    obstacles=None, 
    num_samples=100, 
    horizon=15, 
    alpha=0.1,
    dt=0.1,
    v=1.0
):
    """
    1) Sample many random steering sequences
    2) Roll out each sequence from init_state
    3) Compute cost
    4) Weight them by exp(-alpha * cost)
    5) Return the best or weighted-average control
    """
    control_sequences = np.zeros((num_samples, horizon))
    costs = np.zeros(num_samples)
    all_rollouts = []
    
    for i in range(num_samples):
        # Sample a random sequence of steering angles
        steering_seq = np.random.normal(loc=0.0, scale=0.4, size=horizon)
        control_sequences[i] = steering_seq
        
        # Roll out the sequence
        states = [init_state.copy()]
        current_state = init_state.copy()
        for u in steering_seq:
            current_state = car_dynamics(current_state, u, dt=dt, v=v)
            states.append(current_state)
        
        states = np.array(states)
        all_rollouts.append(states)
        
        # Compute cost (multi-obstacles)
        cost_i = compute_cost(states, goal_state, obstacles=obstacles, penalty=100.0)
        costs[i] = cost_i
    
    all_rollouts = np.array(all_rollouts)  # shape: (num_samples, horizon+1, 3)
    
    # Weight each sequence
    weights = np.exp(-alpha * costs)
    weights /= (np.sum(weights) + 1e-9)  # normalize
    
    # Weighted average of the control sequences
    weighted_controls = np.sum(control_sequences.T * weights, axis=1)
    
    # Identify best sequence (lowest cost)
    best_idx = np.argmin(costs)
    best_sequence = control_sequences[best_idx]
    best_rollout = all_rollouts[best_idx]
    
    return weighted_controls, best_sequence, best_rollout, all_rollouts, costs


# -------------------------
# 4. Run MPPI in a loop
# -------------------------
def run_mppi_control_loop(
    init_state,
    goal_state,
    obstacles=None,
    num_samples=100,
    horizon=15,
    alpha=0.2,
    dt=0.1,
    v=1.0,
    max_iterations=50,
    goal_tolerance=0.2
):
    """
    Run multiple iterations of MPPI until reaching the goal or 
    hitting max_iterations.
    
    Returns:
      - states_history: array of shape (T, 3)
      - controls_history: array of shape (T-1,) with steering angles
    """
    current_state = init_state.copy()
    states_history = [current_state.copy()]
    controls_history = []
    
    for it in range(max_iterations):
        (
            weighted_controls,
            best_sequence,
            best_rollout,
            all_rollouts,
            costs
        ) = mppi_iteration(
            current_state,
            goal_state,
            obstacles=obstacles,
            num_samples=num_samples,
            horizon=horizon,
            alpha=alpha,
            dt=dt,
            v=v
        )
        
        # Pick the first control from the best sequence
        chosen_control = best_sequence[0]
        
        # Apply that control
        current_state = car_dynamics(current_state, chosen_control, dt=dt, v=v)
        states_history.append(current_state.copy())
        controls_history.append(chosen_control)
        
        # Check if we're close enough to the goal
        dist_to_goal = np.linalg.norm(current_state[:2] - goal_state[:2])
        if dist_to_goal < goal_tolerance:
            print(f"Reached goal in {it+1} iterations!")
            break
    
    return np.array(states_history), np.array(controls_history)


# -------------------------
# 5. Animation Demo
# -------------------------
def main_animation_demo():
    np.random.seed(0)
    
    # Initial state
    init_state = np.array([0.0, 0.0, 0.0])
    # Goal state
    goal_state = np.array([5.0, 5.0, 0.0])
    
    # Define multiple obstacles
    obstacles = [
        {"center": np.array([2.5, 2.5]), "radius": 0.8},
        {"center": np.array([4.5, 3.0]), "radius": 0.6},
        {"center": np.array([2.0, 1.0]), "radius": 0.5},
        {"center": np.array([4.0, 1.5]), "radius": 0.5},
        {"center": np.array([3.0, 4.0]), "radius": 0.5},
    ]
    
    # Run MPPI
    states_history, controls_history = run_mppi_control_loop(
        init_state,
        goal_state,
        obstacles=obstacles,
        num_samples=150,
        horizon=15,
        alpha=0.2,
        dt=0.1,
        v=1.0,
        max_iterations=100,
        goal_tolerance=0.3
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.set_title("MPPI Car Animation with Multiple Obstacles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    
    # Plot each obstacle
    for obs in obstacles:
        ox, oy = obs["center"]
        r = obs["radius"]
        circle = plt.Circle((ox, oy), r, color='orange', alpha=0.3)
        ax.add_patch(circle)

    # Plot goal
    ax.plot(goal_state[0], goal_state[1], 'rx', markersize=10, label="Goal")
    # Plot start
    ax.plot(init_state[0], init_state[1], 'go', label="Start")

    # Car trajectory line (will update in animation)
    line, = ax.plot([], [], 'b.-', label="Car Trajectory")
    
    ax.legend()
    
    # Animation update function
    def init_anim():
        line.set_data([], [])
        return (line,)
    
    def update_anim(frame):
        xdata = states_history[:frame+1, 0]
        ydata = states_history[:frame+1, 1]
        line.set_data(xdata, ydata)
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update_anim,
        frames=len(states_history),
        init_func=init_anim,
        interval=200,
        blit=True
    )
    
    plt.show()

    # If you want to save the animation:
    # ani.save("mppi_car_animation_multi_obstacles.gif", writer="pillow")


if __name__ == "__main__":
    main_animation_demo()
