import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import algo.utils as utils

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score
from tensorboardX                         import SummaryWriter

# set your wandb api key here if you want to use wandb for logging
import wandb
from datetime import datetime
import os
os.environ['WANDB_API_KEY'] = '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'

import sys
disable_wandb = False
if sys.gettrace() is not None:  # 检查是否在debugger中
    disable_wandb = True
    print("Debugger detected, wandb disabled")


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    """Evaluates the policy."""
    eval_env = env
    avg_reward = 0.
    avg_ep_len = 0.
    try:
        for episode_idx in range(eval_episodes):
            state, done = eval_env.reset(), False
            ep_reward = 0.
            ep_len = 0
            max_steps = getattr(eval_env, '_max_episode_steps', 1000) # Get max steps if available
            while not done:
                action = policy.select_action(np.array(state), test=True) # Use test=True for deterministic eval
                next_state, reward, done, info = eval_env.step(action)

                ep_reward += reward
                state = next_state
                ep_len += 1
                if ep_len >= max_steps: # Ensure termination if env doesn't handle it
                    done = True
            avg_reward += ep_reward
            avg_ep_len += ep_len
    except Exception as e:
        print(f"[Error] Exception during policy evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0 # Return 0 or handle error appropriately

    avg_reward /= eval_episodes
    avg_ep_len /= eval_episodes

    eval_id = f"Eval-{eval_cnt}" if eval_cnt is not None else "Evaluation"
    print(f"[{eval_id}] Avg Reward over {eval_episodes} episodes: {avg_reward:.3f} (Avg Ep Len: {avg_ep_len:.1f})")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="SAC", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain') # only useful when target domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(4e5), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--eval_freq', default=int(5e3), type=int, help="Evaluation frequency (gradient steps)")
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    
    args = parser.parse_args()

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(f"Domain detected: {domain}")

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if args.mode == 1 or args.mode == 3:
        if domain == 'antmaze':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        elif domain == 'adroit':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        else:
            src_env_name += '-' + args.srctype + '-v2'
        src_env = None
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed)
    else:
        if 'antmaze' in src_env_name:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': None,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
        else:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': None,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = copy.deepcopy(src_env)
            src_eval_env.seed(args.seed + 100)

    if args.mode == 2 or args.mode == 3:
        tar_env = None
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    else:
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_env = call_env[domain](tar_env_config)
        tar_env.seed(args.seed)
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    
    if args.mode not in [0,1,2,3]:
        raise NotImplementedError # cannot support other modes
    
    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    if args.mode == 1:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 2:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 3:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    else:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
        'task_name': src_env_name
    })

    policy = call_algo(args.policy, config, args.mode, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # --- Weights & Biases Initialization ---
    # Initialize wandb to None first, attempt to init, and proceed if it fails
    wandb_instance = None

    # --- Create Output Directory and Logger ---
    config_to_save = config.copy()
    config_to_save['device'] = str(config['device']) 
    config_str = json.dumps(config_to_save, sort_keys=True)

    # Construct descriptive directory name
    run_name_parts = [args.policy, args.env]
    run_name_parts.append(f"src_{args.srctype}")

    run_name_parts.append(f"shift_level_{args.shift_level}")
    if config.get("extreme_shift", False):
        run_name_parts.append("extreme")
    run_name_parts.append(f"seed{args.seed}")
    run_name = "-".join(run_name_parts)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name}_{timestamp}"



    try:
        wandb_run_name = run_name # Use the descriptive run name generated earlier
        if hasattr(policy, 'total_it') and policy.total_it > 0: # If loaded a checkpoint
             wandb_run_name = f"resume_{run_name}"
        if config.get("extreme_shift", False):
            group = f"{args.env}-{args.srctype}-{args.shift_level}-extreme-True"
        else:
            group = f"{args.env}-{args.srctype}-{args.shift_level}"
        if not disable_wandb:
            wandb_instance = wandb.init(
                project="off-dynamics-rl", # Replace with your project name
                entity="zhoukek-zhejiang-university", # Replace with your wandb entity (username or team) or leave as None for default
                name=wandb_run_name,
                config=config, # Log the final config
                dir=outdir, # Save wandb files within the run directory
                resume="allow", # Allow resuming if run_id matches
                id=run_name, # Use base run name as ID for potential resume
                group=group,
            )
            print("Weights & Biases initialized successfully.")

        else:
            print("[debugger] Wandb logging disabled.")

    except Exception as e:
        print(f"Line 465: [Error] Failed to initialize Weights & Biases: {e}")
        print("[Warning] Wandb logging disabled.")
        # wandb_instance remains None

    # in case that the domain is offline, we directly load its offline data
    if args.mode == 1 or args.mode == 3:
        src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
        if 'antmaze' in args.env:
            src_replay_buffer.reward -= 1.0
    
    if args.mode == 2 or args.mode == 3:
        tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
        tar_replay_buffer.convert_D4RL(tar_dataset)
        if 'antmaze' in args.env:
            tar_replay_buffer.reward -= 1.0

    eval_cnt = 0
    
    # eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    # eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)

        # --- Initial Policy Evaluation ---
    eval_cnt = 0
    start_step = 0
    # Only evaluate if not resuming from a later step, or always evaluate?
    # Let's always evaluate the current policy state
    print("\n--- Initial Evaluation (Before Training Loop Starts/Resumes) ---")
    if src_eval_env:
        init_src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=f"InitSrc-{start_step}")
        writer.add_scalar('eval/source_return', init_src_eval_return, global_step=start_step)
        if not disable_wandb:
            if wandb_instance: wandb_instance.log({'eval/source_return': init_src_eval_return}, step=start_step)
    # eval_cnt += 1 # Not really used later, maybe remove
    if tar_eval_env:
        init_tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=f"InitTar-{start_step}")
        init_eval_normalized_score = get_normalized_score(init_tar_eval_return, ref_env_name)
        writer.add_scalar('eval/target_return', init_tar_eval_return, global_step=start_step)
        writer.add_scalar('eval/target_normalized_score', init_eval_normalized_score, global_step=start_step)
        if not disable_wandb:
            if wandb_instance: wandb_instance.log({
                'eval/target_return': init_tar_eval_return,
                'eval/target_normalized_score': init_eval_normalized_score
            }, step=start_step)
    print("-------------------------------------------------------------\n")

    eval_cnt += 1

    if args.mode == 0:
        # online-online learning

        src_state, src_done = src_env.reset(), False
        tar_state, tar_done = tar_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                # record normalized score
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                # record normalized score
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 1:
        # offline-online learning
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)
                if not disable_wandb:
                    if wandb_instance: 
                        wandb_instance.log({
                            'train/target_return': tar_episode_reward,
                            'train/target_normalized_score': train_normalized_score,
                        }, step=t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
                if not disable_wandb:
                    if wandb_instance: wandb_instance.log({
                            'test/source_return': src_eval_return,
                            'test/target_return': tar_eval_return,
                            'test/target_normalized_score': eval_normalized_score,
                        }, step=t+1)   

                eval_cnt += 1

                if (t + 1) == 400000:
                    tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=f"Step{t+1}-Tar")
                    eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                    if not disable_wandb:
                        if wandb_instance: wandb_instance.log({
                            'test/target_return_400K': tar_eval_return,
                            'test/target_normalized_score_400K': eval_normalized_score,
                        })    

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 2:
        # online-offline learning
        src_state, src_done = src_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    else:
        # offline-offline learning
        for t in range(int(config['max_step'])):
            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
                
                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    writer.close()
