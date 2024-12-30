#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque
from obs_process import obs_process, get_process_obs_dim

import utils
import hydra


os.environ["MUJOCO_GL"] = "osmesa"


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
            )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        if 'metaworld' in self.cfg.env:
            self.env = utils.make_metaworld_env(self.cfg)
            self.env.reset()
            self.log_success = True
        elif 'PointMaze' in self.cfg.env:
            import gymnasium as gym
            import gymnasium_robotics
            gym.register_envs(gymnasium_robotics)
            self.env = gym.make(self.cfg.env, max_episode_steps=100)
            self.log_success = True
        else:
            self.env = utils.make_env(self.cfg)
        
        obs = self.env.reset()
        if "metaworld" in self.cfg.env:
            obs = obs[0]
        elif 'PointMaze' in self.cfg.env:
            obs = np.concatenate((obs[0]['observation'], obs[0]['desired_goal']))
            # obs_dim_for_reward = 4

        process_obs_shape = get_process_obs_dim(self.cfg.env, obs, self.cfg.process_type)
        self.cfg.agent.params.obs_dim = process_obs_shape[0]
        self.cfg.agent.params.action_dim = self.env.action_space.shape[0]
        self.cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(self.cfg.agent)

        self.replay_buffer = ReplayBuffer(
            process_obs_shape,
            self.env.action_space.shape,
            int(self.cfg.replay_buffer_capacity),
            self.cfg.traj_action,
            self.device)
        
        traj_save_path = f"{self.work_dir}/{self.cfg.traj_save_path}" if self.cfg.traj_save_path is not None else None
        # instantiating the reward model
        self.reward_model = RewardModel(
            process_obs_shape[0],
            # obs_dim_for_reward,
            self.env.action_space.shape[0],
            ensemble_size=self.cfg.ensemble_size,
            size_segment=self.cfg.segment,
            activation=self.cfg.activation, 
            lr=self.cfg.reward_lr,
            mb_size=self.cfg.reward_batch, 
            large_batch=self.cfg.large_batch, 
            label_margin=self.cfg.label_margin, 
            teacher_beta=self.cfg.teacher_beta, 
            teacher_gamma=self.cfg.teacher_gamma, 
            teacher_eps_mistake=self.cfg.teacher_eps_mistake, 
            teacher_eps_skip=self.cfg.teacher_eps_skip, 
            teacher_eps_equal=self.cfg.teacher_eps_equal,
            env_name=self.cfg.env,
            traj_action=self.cfg.traj_action,
            traj_save_path=traj_save_path,
            vlm_label=self.cfg.vlm_label,
            better_traj_gen=self.cfg.better_traj_gen,
            double_check=self.cfg.double_check,
            save_equal=self.cfg.save_equal,
            vlm_feedback=self.cfg.vlm_feedback,
            generate_check=self.cfg.generate_check)
        
        self.step = 0
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.llm_query_accuracy = 0
        self.llm_label_accuracy = 0

        # Check if there is a checkpoint to resume from
        self.checkpoint_path = os.path.join(self.work_dir, 'checkpoint.pth')
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
        # else:
            # Initialize environment, agent, replay buffer, and reward model from scratch
            # self._initialize_from_scratch()
            

    # def _initialize_from_scratch(self):
    #     # make env

    def save_checkpoint(self):
        checkpoint = {
            'step': self.step,
            'replay_buffer': self.replay_buffer,
            'total_feedback': self.total_feedback,
            'labeled_feedback': self.labeled_feedback,
            'llm_query_accuracy': self.llm_query_accuracy,
            'llm_label_accuracy': self.llm_label_accuracy
        }
        with open(self.checkpoint_path, 'wb') as f:
            pkl.dump(checkpoint, f)
        print(f'Checkpoint saved at step {self.step}')

        # Save agent and reward model separately
        self.agent.save(self.work_dir, 'latest')
        self.reward_model.save(self.work_dir, 'latest')


    def load_checkpoint(self):
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pkl.load(f)
        self.step = checkpoint['step']
        self.replay_buffer = checkpoint['replay_buffer']
        self.total_feedback = checkpoint['total_feedback']
        self.labeled_feedback = checkpoint['labeled_feedback']
        self.llm_query_accuracy = checkpoint['llm_query_accuracy']
        self.llm_label_accuracy = checkpoint['llm_label_accuracy']
        print(f'Resuming from checkpoint at step {self.step}')

        
        # Load agent and reward model separately
        self.agent.load(self.work_dir, 'latest')
        self.reward_model.load(self.work_dir, 'latest')


    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        # self.env.camera_name="corner3"

        for episode in range(self.cfg.num_eval_episodes):
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]
            elif 'PointMaze' in self.cfg.env:
                obs = np.concatenate((obs[0]['observation'], obs[0]['desired_goal']))
            obs = obs_process(self.cfg.env, obs, self.cfg.process_type)
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                # obs, reward, done, extra = self.env.step(action)
                try: # for handle stupid gym wrapper change 
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated

                if 'PointMaze' in self.cfg.env:
                    obs = np.concatenate((obs['observation'], obs['desired_goal']))

                obs = obs_process(self.cfg.env, obs, self.cfg.process_type)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

                if self.cfg.save_video and ("metaworld" in self.cfg.env):
                    
                    rgb_image = self.env.render()
                    rgb_image = rgb_image[::-1, :, :]

                    images.append(rgb_image)

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
            if self.cfg.save_video:
                save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2)))
                utils.save_numpy_as_gif(np.array(images), save_gif_path)
            

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            # self.logger.log('train/true_episode_success', success_rate,
            #             self.step)
        self.logger.dump(self.step)

        self.env.close()
    
    def learn_reward(self, first_flag=0):            
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0 or ((self.cfg.vlm_label is not None) and (self.cfg.save_equal)):
                    self.llm_query_accuracy, llm_label_accuracy, train_acc = self.reward_model.train_soft_reward()
                else:
                    self.llm_query_accuracy, llm_label_accuracy, train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                self.llm_label_accuracy = np.mean(llm_label_accuracy)
                
                if total_acc > 0.97:
                    break
                    
        print("Reward function is updated!! ACC: " + str(total_acc))
        print("Reward function is updated!! LLM LABEL ACC: " + str(self.llm_label_accuracy))


    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        first_evaluate = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    
                    # 这边第一次不要evaluate，否则可能有bug，对于断点续传的程序
                    # if first_evaluate == 1 or (not os.path.exists(self.checkpoint_path)):
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    # elif first_evaluate == 0:
                    #     first_evaluate = 1
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                self.logger.log('train/llm_query_accuracy', self.llm_query_accuracy, self.step)
                self.logger.log('train/llm_label_accuracy', self.llm_label_accuracy, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                if "metaworld" in self.cfg.env:
                    obs = obs[0]
                elif 'PointMaze' in self.cfg.env:
                    obs = np.concatenate((obs[0]['observation'], obs[0]['desired_goal']))
                obs = obs_process(self.cfg.env, obs, self.cfg.process_type)

                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            try:
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated

            if 'PointMaze' in self.cfg.env:
                next_obs = np.concatenate((next_obs['observation'], next_obs['desired_goal']))
            next_obs = obs_process(self.cfg.env, next_obs, self.cfg.process_type)

            if self.cfg.traj_action:
                reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            else:
                # reward_hat = self.reward_model.r_hat(obs)
                obs_reward = np.concatenate([obs[0:2], obs[4:6]], axis=-1)
                reward_hat = self.reward_model.r_hat(obs_reward)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            # self.reward_model.add_data(obs, action, reward, done)
            self.reward_model.add_data(obs_reward, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.step % 10000 == 0:
                self.save_checkpoint()
        
@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()
  
if __name__ == '__main__':
    main()