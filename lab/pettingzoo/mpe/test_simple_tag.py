from pettingzoo.mpe import simple_tag_v3

if __name__ == '__main__':
    env = simple_tag_v3.parallel_env(render_mode="human", num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=50,
                                     continuous_actions=False)
    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
