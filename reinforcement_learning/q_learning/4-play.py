import numpy as np

def play(env, Q, max_steps=100):
    """Have the trained agent play one episode; render each step and return total reward."""
    # Handle gym vs gymnasium reset signatures
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    total_reward = 0.0
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    for _ in range(max_steps):
        # Show the board *before* taking the action
        env.render()

        # Greedy action from Q-table
        action = int(np.argmax(Q[state, :]))
        print(f"  ({action_names.get(action, str(action))})")

        # Step (handle both 4-tuple and 5-tuple APIs)
        step_out = env.step(action)
        if len(step_out) == 5:
            new_state, reward, terminated, truncated, _info = step_out
            done = terminated or truncated
        else:
            new_state, reward, done, _info = step_out

        total_reward += float(reward)
        state = new_state

        if done:
            # Final board after reaching terminal state
            env.render()
            break

    env.close()
    return total_reward
