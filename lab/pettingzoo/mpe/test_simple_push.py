from pettingzoo.mpe import simple_push_v3

if __name__ == '__main__':
    env = simple_push_v3.parallel_env(max_cycles=25, render_mode="human")
    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
