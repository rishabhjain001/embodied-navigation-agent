import math


def euclidean_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_reward(
    action: int,
    prev_agent_pos: tuple[int, int],
    curr_agent_pos: tuple[int, int],
    target_pos: tuple[int, int],
    step_count: int,
    step_budget: int,
    done_stop: bool,
) -> tuple[float, bool]:
    reward = 0.0
    done = False

    if action == 0:  # move_forward
        prev_dist = euclidean_distance(prev_agent_pos, target_pos)
        curr_dist = euclidean_distance(curr_agent_pos, target_pos)
        if prev_dist > curr_dist:
            reward += 0.1
        elif prev_dist < curr_dist:
            reward += -0.1
        # collision: forward was blocked, position unchanged
        if curr_agent_pos == prev_agent_pos:
            reward += -0.5

    elif action == 3:  # stop
        if manhattan_distance(curr_agent_pos, target_pos) <= 1:
            reward += 10.0
        done = True

    # timeout check applies regardless of action
    if step_count >= step_budget:
        reward += -5.0
        done = True

    return reward, done
