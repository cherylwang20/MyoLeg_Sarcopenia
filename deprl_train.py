import gym
import myosuite
import deprl

# we can also change the reset_type of the environment here
env = gym.make('myoLegWalk-v0', reset_type='random')
policy = deprl.load_baseline(env)

for ep in range(5):
    obs = env.reset()
    for i in range(1000):
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        env.sim.renderer.render_to_window()
        obs = next_obs
        if done:
            break