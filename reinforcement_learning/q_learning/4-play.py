#!/usr/bin/env python3
import re
import numpy as np

_ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
_ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def _call_render(env):
    """Call render for gymnasium or classic gym."""
    try:
        out = env.render()  # gymnasium: returns str with render_mode="ansi"
    except TypeError:
        out = None
    if out is None:
        try:
            out = env.render(mode="ansi")  # classic gym fallback
        except Exception:
            out = ""
    # Some gyms may return StringIO
    if hasattr(out, "getvalue"):
        out = out.getvalue()
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="ignore")
    return out if isinstance(out, str) else str(out)

def _normalize_board_str(s):
    """Strip ANSI codes, convert backticks to quotes, trim trailing newlines."""
    if not isinstance(s, str):
        s = str(s)
    s = _ANSI_RE.sub("", s)
    s = s.replace("`", '"')
    return s.rstrip()

def play(env, Q, max_steps=100):
    """
    Plays one episode by always exploiting the Q-table.

    Returns:
        total_rewards (float), rendered_outputs (list[str])
    """
    rendered_outputs = []
    total_rewards = 0.0

    # Reset (gymnasium returns (obs, info); classic gym returns obs)
    try:
        state, _ = env.reset()
    except Exception:
        state = env.reset()

    # Initial board
    rendered_outputs.append(_normalize_board_str(_call_render(env)))

    for _ in range(max_steps):
        # Exploit best action
        action = int(np.argmax(Q[state]))

        step = env.step(action)
        if len(step) == 5:
            next_state, reward, terminated, truncated, _info = step
            done = terminated or truncated
        else:
            next_state, reward, done, _info = step

        total_rewards += float(reward)

        # Action line
        rendered_outputs.append(f"  ({_ACTION_NAMES.get(action, str(action))})")

        # Board after action (ensure final state is shown)
        rendered_outputs.append(_normalize_board_str(_call_render(env)))

        state = next_state
        if done:
            break

    return total_rewards, rendered_outputs
