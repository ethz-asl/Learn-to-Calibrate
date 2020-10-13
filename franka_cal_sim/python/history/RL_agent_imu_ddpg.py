#!/usr/bin/env python


from ray import tune
from ray.rllib.agents.ddpg import DDPGTrainer
import random
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from calibration_env import CamCalibrEnv, imuCalibrEnv_seq
import ray

import psutil
import gc

#model defination (hidden state output)
class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)



        #input
        self.input = tf.keras.layers.Input(shape=(None,48))

        #network (recurrent model out)
        mask_state_input = tf.keras.layers.Masking(mask_value=0.)(self.input)
        obs_h1 = tf.keras.layers.Dense(256, activation='relu')(mask_state_input)
        actor_rnn,state_h = tf.keras.layers.GRU(256, return_state=True)(obs_h1)
        self.output = state_h

        self.base_model = tf.keras.Model(self.input, self.output)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state


#init
ray.init()

#model registration
ModelCatalog.register_custom_model("my_model", MyKerasModel)

config = {
#config
# === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
# TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
# In addition to settings below, you can use "exploration_noise_type" and
# "exploration_gauss_act_noise" to get IID Gaussian exploration noise
# instead of OU exploration noise.
# twin Q-net
"twin_q": True,
# delayed policy update
"policy_delay": 1,
# target policy smoothing
# (this also replaces OU exploration noise with IID Gaussian exploration
# noise, for now)
"smooth_target_policy": True,
# gaussian stddev of target action noise for smoothing
"target_noise": 0.2,
# target noise limit (bound)
"target_noise_clip": 0.5,

# === Evaluation ===
# Evaluate with epsilon=0 every `evaluation_interval` training iterations.
# The evaluation stats will be reported under the "evaluation" metric key.
# Note that evaluation is currently not parallelized, and that for Ape-X
# metrics are already only reported for the lowest epsilon workers.
"evaluation_interval":10,
# Number of episodes to run per evaluation period.
"evaluation_num_episodes": 1,

# === Model ===
# Apply a state preprocessor with spec given by the "model" config option
# (like other RL algorithms). This is mostly useful if you have a weird
# observation shape, like an image. Disabled by default.
"use_state_preprocessor": True,
# Postprocess the policy network model output with these hidden layers. If
# use_state_preprocessor is False, then these will be the *only* hidden
# layers in the network.
"actor_hiddens": [512],
# Hidden layers activation of the postprocessing stage of the policy
# network
"actor_hidden_activation": "relu",
# Postprocess the critic network model output with these hidden layers;
# again, if use_state_preprocessor is True, then the state will be
# preprocessed by the model specified with the "model" config option first.
"critic_hiddens": [512],
# Hidden layers activation of the postprocessing state of the critic.
"critic_hidden_activation": "relu",
# N-step Q learning
"n_step": 3,

"horizon": 3,

# === Exploration ===
"exploration_config": {
    # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
    # actions (after a possible pure random phase of n timesteps).
    "type": "OrnsteinUhlenbeckNoise",
    # For how many timesteps should we return completely random actions,
    # before we start adding (scaled) noise?
    "random_timesteps": 100,
    # The OU-base scaling factor to always apply to action-added noise.
    "ou_base_scale": 0.1,
    # The OU theta param.
    "ou_theta": 0.15,
    # The OU sigma param.
    "ou_sigma": 0.2,
    # The initial noise scaling factor.
    "initial_scale": 1.0,
    # The final noise scaling factor.
    "final_scale": 0.1,
    # Timesteps over which to anneal scale (from initial to final values).
    "scale_timesteps": 2000,
},
# Number of env steps to optimize for before returning
"timesteps_per_iteration": 20,
# If True parameter space noise will be used for exploration
# See https://blog.openai.com/better-exploration-with-parameter-noise/
"parameter_noise": False,
# Extra configuration that disables exploration.
"evaluation_config": {
    "explore": False
},
# === Replay buffer ===
# Size of the replay buffer. Note that if async_updates is set, then
# each worker will have a replay buffer of this size.
"buffer_size": 4000,
# If True prioritized replay buffer will be used.
"prioritized_replay": True,
# Alpha parameter for prioritized replay buffer.
"prioritized_replay_alpha": 0.6,
# Beta parameter for sampling from prioritized replay buffer.
"prioritized_replay_beta": 0.4,
# Time steps over which the beta parameter is annealed.
"prioritized_replay_beta_annealing_timesteps": 400,
# Final value of beta
"final_prioritized_replay_beta": 0.4,
# Epsilon to add to the TD errors when updating priorities.
"prioritized_replay_eps": 1e-6,
# Whether to LZ4 compress observations
"compress_observations": False,
"clip_actions": False,

"normalize_actions": False,

# === Optimization ===
# Learning rate for the critic (Q-function) optimizer.
"critic_lr": 1e-6,
# Learning rate for the actor (policy) optimizer.
"actor_lr": 1e-6,
# Update the target network every `target_network_update_freq` steps.
"target_network_update_freq": 0,
# Update the target by \tau * policy + (1-\tau) * target_policy
"tau": 0.01,
# If True, use huber loss instead of squared loss for critic network
# Conventionally, no need to clip gradients if using a huber loss
"use_huber": False,
# Threshold of a huber loss
"huber_threshold": 1.0,
# Weights for L2 regularization
"l2_reg": 1e-6,
# If not None, clip gradients during optimization at this value
"grad_norm_clipping": None,
# How many steps of the model to sample before learning starts.
"learning_starts": 100,
# Update the replay buffer with this many samples at once. Note that this
# setting applies per-worker if num_workers > 1.
"rollout_fragment_length": 1,
# Size of a batched sampled from replay buffer for training. Note that
# if async_updates is set, then each worker returns gradients for a
# batch of this size.
"train_batch_size": 64,

# === Parallelism ===
# Number of workers for collecting samples with. This only makes sense
# to increase if your environment is particularly slow to sample, or if
# you're using the Async or Ape-X optimizers.
"num_workers": 0,
# Whether to compute priorities on workers.
"worker_side_prioritization": False,
# Prevent iterations from going lower than this time span
"min_iter_time_s": 1,


"model":{
    "custom_model": "my_model",
},
}

def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

#trainable function
def train(config, reporter):
    trainer = DDPGTrainer(config=config, env=imuCalibrEnv_seq)
    #checkpoint_path = trainer.save()
    policy = trainer.get_policy()
    print(policy.dist_class)

    i = 0
    while True:
        result = trainer.train()
        reporter(**result)
        # if result["timesteps_since_restore"] > 200:
        #     phase = 1
        # else:
        #     phase = 0
        # trainer.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_phase(phase)))
        # if i==0:
        #     trainer.restore("/home/yunke/ray_results/DDPG_imuCalibrEnv_seq_2020-06-27_01-48-53hwk9uq89/checkpoint_995/checkpoint-995")
        if i>3 and i%100==0:
            checkpoint_path = trainer.save()
            print(checkpoint_path)
        auto_garbage_collect()
        i+=1
#train
analysis=tune.run(train,
                  stop={"timesteps_total": 200000},
                  name="cal_ddpg_test_new",
                  config = config
                  )
