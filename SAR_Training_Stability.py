from SAR_tutorial_utils import *
from stable_baselines3 import PPO, SAC

from datetime import datetime
import torch

def train(env_name, policy_name, timesteps, seed):
    """
    Trains a policy using sb3 implementation of SAC.
    
    env_name: str; name of gym env.
    policy_name: str; choose unique identifier of this policy
    timesteps: int; how long you want to train your policy for
    seed: str (not int); relevant if you want to train multiple policies with the same params
    """
    env = gym.make(env_name)
    #env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))
    model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
            learning_starts=1000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
            gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1)
    succ_callback = SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
                             log_dir=f'{policy_name}_successes_myoLegReachFixed-v2_{seed}')
    
    model.set_logger(configure(f'{policy_name}_results_myoLegReachFixed-v2_{seed}'))
    model.learn(total_timesteps=int(timesteps), callback=succ_callback, log_interval=4)
    model.save(f"{policy_name}_model_myoLegReachFixed-v2_{seed}")
    env.save(f'{policy_name}_env_myoLegReachFixed-v2_{seed}')

def SAR_RL(env_name, policy_name, timesteps, seed, ica, pca, normalizer, phi=.66, syn_nosyn=True):
    """
    Trains a policy using sb3 implementation of SAC + SynNoSynWrapper.
    
    env_name: str; name of gym env.
    policy_name: str; choose unique identifier of this policy
    timesteps: int; how long you want to train your policy for
    seed: str (not int); relevant if you want to train multiple policies with the same params
    ica: the ICA object
    pca: the PCA object
    normalizer: the normalizer object
    phi: float; blend parameter between synergistic and nonsynergistic activations
    """
    if syn_nosyn:
        env = SynNoSynWrapper(gym.make(env_name), ica, pca, normalizer, phi)
    else:
        env = SynergyWrapper(gym.make(env_name), ica, pca, normalizer)
    #env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))

    
    model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
            learning_starts=5000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
            gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1)
    
    succ_callback = SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
                             log_dir=f'./standingBalance/{policy_name}_successes_myoLegReachFixed-v2_{seed}')
    
    model.set_logger(configure(f'{policy_name}_results_myoLegReachFixed-v2_{seed}'))
    model.learn(total_timesteps=int(timesteps), callback=succ_callback, log_interval=4)
    model.save(f"./StandingBalance/{policy_name}_model_myoLegReachFixed-v2_{seed}")
    env.save(f'./StandingBalance/{policy_name}_env_myoLegReachFixed-v2_{seed}')
def get_activations(name, env_name, seed, episodes=2000, percentile=80):
    """
    Returns muscle activation data from N runs of a trained policy.

    name: str; policy name (see train())
    env_name: str; name of the gym environment
    seed: str; seed of the trained policy
    episodes: int; optional; how many rollouts?
    percentile: int; optional; percentile to set the reward threshold for considering an episode as successful
    """
    with gym.make(env_name) as env:
        env.reset()

        model = SAC.load(f'{name}_model_myoLegReachFixed-v2_{seed}')
        vec = VecNormalize.load(f'{name}_env_myoLegReachFixed-v2_{seed}', DummyVecEnv([lambda: env]))

        # Calculate the reward threshold from 100 preview episodes
        preview_rewards = []
        for _ in range(100):
            env.reset()
            rewards = 0
            done = False
            while not done:
                o = env.get_obs()
                o = vec.normalize_obs(o)
                a, __ = model.predict(o, deterministic=False)
                next_o, r, done, info = env.step(a)
                rewards += r
            preview_rewards.append(rewards)
        reward_threshold = np.percentile(preview_rewards, percentile)

        # Run the main episode loop
        solved_acts = []
        for _ in tqdm(range(episodes)):
            env.reset()
            rewards, acts = 0, []
            done = False

            while not done:
                o = env.get_obs()
                o = vec.normalize_obs(o)
                a, __ = model.predict(o, deterministic=False)
                next_o, r, done, info = env.step(a)
                acts.append(env.sim.data.act.copy())
                rewards += r

            if rewards > reward_threshold:
                solved_acts.extend(acts)

    return solved_acts

def find_synergies(acts, plot=True):
    """
    Computed % variance explained in the original muscle activation data with N synergies.
    
    acts: np.array; rollout data containing the muscle activations
    plot: bool; whether to plot the result
    """
    syn_dict = {}
    for i in range(len(acts[1])):
        pca = PCA(n_components=i+1)
        _ = pca.fit_transform(acts)
        syn_dict[i+1] =  round(sum(pca.explained_variance_ratio_), 4)
        
    if plot:
        plt.plot(list(syn_dict.keys()), list(syn_dict.values()))
        plt.title('VAF by N synergies')
        plt.xlabel('# synergies')
        plt.ylabel('VAF')
        plt.grid()
        plt.show()
    return syn_dict

def compute_SAR(acts, n_syn, save=False):
    """
    Takes activation data and desired n_comp as input and returns/optionally saves the ICA, PCA, and Scaler objects
    
    acts: np.array; rollout data containing the muscle activations
    n_comp: int; number of synergies to use
    """
    pca = PCA(n_components=n_syn)
    pca_act = pca.fit_transform(acts)
    
    ica = FastICA()
    pcaica_act = ica.fit_transform(pca_act)
    
    normalizer = MinMaxScaler((-1, 1))    
    normalizer.fit(pcaica_act)
    
    if save:
        joblib.dump(ica, 'ica_demo.pkl') 
        joblib.dump(pca, 'pca_demo.pkl')
        joblib.dump(normalizer, 'scaler_demo.pkl')
    
    return ica, pca, normalizer


env_name = 'mj_envs.robohive.envs.myo:myoLegReachFixed-v2'

'''
train(env_name, 'play_period', 1.5e3, '0')
muscle_data = get_activations(name='play_period', env_name=env_name, seed='0', episodes=10)
syn_dict = find_synergies(muscle_data, plot=False)
ica,pca,normalizer = compute_SAR(muscle_data, 20, save=False)
'''
ica,pca,normalizer = load_locomotion_SAR()
SAR_RL(env_name = env_name ,  policy_name = 'SAR_RL_Leg_Stability', timesteps = 5e3, seed = '0', ica=ica, pca=pca, normalizer = normalizer, phi = 0.66, syn_nosyn=False)

video_name = 'stability_SAR-RL_video'

get_vid(name='SAR_RL_Leg_Stability', env_name=env_name, seed='0', episodes=5, video_name=video_name, 
        determ=False, pca=pca, ica=ica, normalizer=normalizer, phi=.66, is_sar=True, syn_nosyn=False)

show_video(f"{video_name}.mp4")