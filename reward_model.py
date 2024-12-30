import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
import json

from scipy.stats import norm

from obs_process import traj_pair_process, format_traj_for_gpt, get_response_answer, extract_trajectory_from_text, convert_trajdist_to_traj, format_traj_begin_for_gpt
from prompt import (
    gpt_query_env_prompts,
    trajgen_template,
)

device = 'cuda'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 env_name=None,
                 traj_action=True,
                 traj_save_path=None,
                 vlm_label=None,
                 better_traj_gen=False,
                 double_check=False,
                 save_equal=True,
                 vlm_feedback=True,
                 generate_check=False,
                 ):
    
        # train data is trajectories, must process to sa and s..   
        self.env_name = env_name
        self.traj_action = traj_action
        self.traj_save_path = traj_save_path

        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.vlm_label = vlm_label
        self.better_traj_gen = better_traj_gen
        self.double_check = double_check
        self.save_equal = save_equal
        self.vlm_feedback = vlm_feedback
        self.generate_check = generate_check
        
        self.capacity = int(capacity)
        if traj_action:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        else:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32)
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32)

        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.gt_buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.fake_flag = np.empty((self.capacity, 1), dtype=bool)

        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

        # for llms
        self.llm_query_accuracy = 0
        self.llm_label_accuracy = 0

        

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            if self.traj_action:
                model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            else:
                model = nn.Sequential(*gen_net(in_size=self.ds, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        if self.traj_action:
            sa_t = np.concatenate([obs, act], axis=-1)
            flat_input = sa_t.reshape(1, self.da+self.ds)
        else:
            sa_t = obs
            flat_input = sa_t.reshape(1, self.ds)

        r_t = rew
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            gt_labels = self.gt_buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)

            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                gt_correct = (predicted == gt_labels).sum().item()
                ensemble_acc[member] += correct
                gt_ensemble_acc[member] += gt_correct
                
        ensemble_acc = ensemble_acc / total
        gt_ensemble_acc = gt_ensemble_acc / total
        return np.mean(ensemble_acc), np.mean(gt_ensemble_acc)
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels, gt_labels, r_t_1=None, r_t_2=None, fake_traj=False):
        total_sample = sa_t_1.shape[0]
        origin_index = self.buffer_index
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            np.copyto(self.gt_buffer_label[self.buffer_index:self.capacity], gt_labels[:maximum_index])
            self.fake_flag[self.buffer_index:self.capacity] = fake_traj

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                np.copyto(self.gt_buffer_label[0:remain], gt_labels[maximum_index:])
                self.fake_flag[0:remain] = fake_traj

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            np.copyto(self.gt_buffer_label[self.buffer_index:next_index], gt_labels)
            self.fake_flag[self.buffer_index:next_index] = fake_traj
            self.buffer_index = next_index
        


    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        origin_index = self.buffer_index
        # if not self.vlm_label:
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                            torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1 

        gt_labels = labels 
        # else:
        if self.vlm_label:
            labels = self.get_vlm_label(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels
    

    def get_vlm_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, labels):
        origin_index = self.buffer_index

        from vlms.gpt4_infer import gpt_infer, gpt_infer_traj_gen, gpt_infer_traj_gen_check

        # Create an array with the same shape as `labels`, initialized with -1
        vlm_labels = np.full(labels.shape, -1, dtype=int)

        for idx in range(sa_t_1.shape[0]):
            print("querying {} {}/{}".format(self.vlm_label, idx, sa_t_1.shape[0]))
            data_to_save, traj_1, traj_2 = traj_pair_process(self.env_name, sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], labels[idx])
            traj_str = format_traj_for_gpt(traj_1, traj_2)
            query_prompt = gpt_query_env_prompts[self.env_name]
            query_prompt = query_prompt.format(self.size_segment, traj_str)

            # Call GPT inference and get the answer
            res = gpt_infer(self.vlm_label, query_prompt)
            answer = get_response_answer(res)

            try:
                # Try to convert the answer to an integer
                label_res = int(answer)
                # If `label_res` is 1 or 2, subtract 1
                if label_res == 1 or label_res == 2:
                    label_res = label_res - 1
                else:
                    # Otherwise, set it to -1
                    label_res = -1
            except:
                # If any exception occurs, set it to -1
                label_res = -1

            if self.double_check and label_res != -1:
                # Perform double-check logic by swapping `traj_1` and `traj_2`
                traj_str_swapped = format_traj_for_gpt(traj_2, traj_1)
                query_prompt_swapped = gpt_query_env_prompts[self.env_name].format(self.size_segment, traj_str_swapped)

                # Call GPT inference again
                res_swapped = gpt_infer(self.vlm_label, query_prompt_swapped)
                answer_swapped = get_response_answer(res_swapped)

                try:
                    # Try to convert the swapped answer to an integer
                    label_res_swapped = int(answer_swapped)
                    # If `label_res_swapped` is 1 or 2, subtract 1
                    if label_res_swapped == 1 or label_res_swapped == 2:
                        label_res_swapped = label_res_swapped - 1
                    else:
                        label_res_swapped = -1
                except:
                    label_res_swapped = -1

                # If the two predictions are not symmetric, discard the label and set it to -1
                if not ((label_res == 0 and label_res_swapped == 1) or (label_res == 1 and label_res_swapped == 0)):
                    print("Double check False!")
                    label_res = -1

            # Fill the result into the corresponding position of `vlm_labels`
            vlm_labels[idx] = label_res

            # Save `data_to_save` and `res` to JSON files
            if self.traj_save_path is not None:
                # Create directory if it doesn't exist
                os.makedirs(self.traj_save_path, exist_ok=True)
                # Save `data_to_save`
                data_file_path = os.path.join(self.traj_save_path, f"traj_pairs_{origin_index+idx}.json")
                with open(data_file_path, 'w') as data_file:
                    json.dump(data_to_save, data_file, indent=4)
                # Save `res` (inference results)
                res_file_path = os.path.join(self.traj_save_path, f"llm_response_{origin_index+idx}.json")
                with open(res_file_path, 'w') as res_file:
                    json.dump(res, res_file, default=lambda o: o.__dict__, indent=4)

            if self.better_traj_gen and (label_res != -1):
                print("querying better_traj_gen {} {}/{}".format(self.vlm_label, idx, sa_t_1.shape[0]))
                # Select the better trajectory
                traj_better = traj_1 if label_res == 0 else traj_2

                if "metaworld" in self.env_name:
                    traj_better_begin = {
                        'tcp': traj_better['tcp'][0],
                        'obj': traj_better['obj'][0],
                        'target': traj_better['target']  # Keep all `target` elements
                    }
                elif "PointMaze" in self.env_name:
                    traj_better_begin = {
                        'position': traj_better['position'][0],
                        'target': traj_better['target']  # Keep all `target` elements
                    }

                traj_better_begin_str = format_traj_begin_for_gpt(traj_better_begin)
                trajgen_prompt = trajgen_template(self.env_name).format(self.size_segment, traj_better_begin_str)
                res_traj = gpt_infer_traj_gen(self.vlm_label, query_prompt, res, trajgen_prompt)
                new_traj_dist = extract_trajectory_from_text(self.env_name, res_traj, self.size_segment)

                # Save `res_traj` (inference results)
                if self.traj_save_path is not None:
                    res_traj_file_path = os.path.join(self.traj_save_path, f"llm_trajgen_{origin_index+idx}.json")
                    with open(res_traj_file_path, 'w') as res_traj_file:
                        json.dump(res_traj, res_traj_file, default=lambda o: o.__dict__, indent=4)

                    # Select the worse trajectory
                    worse_traj = sa_t_1[idx] if label_res == 0 else sa_t_2[idx]
                    better_traj = convert_trajdist_to_traj(self.env_name, worse_traj, new_traj_dist)

                    # Reshape to add one dimension
                    worse_traj = worse_traj.reshape(1, *worse_traj.shape)
                    better_traj = better_traj.reshape(1, *better_traj.shape)
                    self.put_queries(worse_traj, better_traj, labels=np.array([[1]]), gt_labels=np.array([[1]]), fake_traj=True)

                    print("Valid trajectory, Store it to buffer...")
                else:
                    print("Invalid trajectory, discarding...")

        # Filter out parts of `vlm_labels` with -1
        valid_indices = [i for i, label in enumerate(vlm_labels) if label != -1]
        if valid_indices:  # If there are valid predictions
            filtered_vlm_labels = vlm_labels[valid_indices]
            filtered_labels = labels[valid_indices]
            # Calculate accuracy
            correct = sum(1 for vlm_label, target_label in zip(filtered_vlm_labels, filtered_labels) if vlm_label == target_label)
            accuracy = correct / len(filtered_vlm_labels)
            print(f"LLM Label Accuracy: {accuracy * 100:.2f}%")
        else:
            print("No valid predictions to compute accuracy.")
            accuracy = 0.0  # If no valid predictions, set accuracy to 0

        self.llm_query_accuracy = accuracy

        return vlm_labels


    

    def kcenter_sampling(self):
        
        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            if self.traj_save_path is not None:
                self.put_queries(sa_t_1, sa_t_2, labels, gt_labels, r_t_1, r_t_2)
            else:
                self.put_queries(sa_t_1, sa_t_2, gt_labels, labels)
        
        return len(labels)
    
    def kcenter_disagree_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            if self.traj_save_path is not None:
                self.put_queries(sa_t_1, sa_t_2, labels, gt_labels, r_t_1, r_t_2)
            else:
                self.put_queries(sa_t_1, sa_t_2, gt_labels, labels)
        
        return len(labels)
    
    def kcenter_entropy_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if not self.save_equal:
                valid_indices = labels[:, 0] != -1
                sa_t_1 = sa_t_1[valid_indices]
                sa_t_2 = sa_t_2[valid_indices]
                r_t_1 = r_t_1[valid_indices]
                r_t_2 = r_t_2[valid_indices]
                labels = labels[valid_indices]
                gt_labels = gt_labels[valid_indices]

        if len(labels) > 0:
            if not (self.vlm_label and  (not self.vlm_feedback) and self.better_traj_gen):
                if self.traj_save_path is not None:
                    self.put_queries(sa_t_1, sa_t_2, labels, gt_labels, r_t_1, r_t_2)
                else:
                    self.put_queries(sa_t_1, sa_t_2, gt_labels, labels)
        
        return len(labels)
    
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]        
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []

        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0
        gt_total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            # Calculate label accuracy only for valid indices (self.fake_flag is False)

            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                gt_labels = self.gt_buffer_label[idxs]
                fake_labels = self.fake_flag[idxs]

                labels = torch.from_numpy(labels.flatten()).long().to(device)
                gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)

                # Note: Do not filter during training, only for label accuracy calculation
                valid_indices = np.where(fake_labels.flatten() == False)[0]

                if member == 0:
                    total += labels.size(0)
                    gt_total += len(valid_indices)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc for all data
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

                
                # Accumulate accuracy on all data
                if len(valid_indices) > 0:
                    gt_correct = (predicted[valid_indices] == gt_labels[valid_indices]).sum().item()
                    gt_ensemble_acc[member] += gt_correct
                else:
                    gt_ensemble_acc[member] += 0

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total
        if gt_total != 0:
            self.llm_label_accuracy = gt_ensemble_acc / gt_total
        else:
            self.llm_label_accuracy = 0 

        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc

    

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                gt_labels = self.gt_buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                gt_correct = (predicted == gt_labels).sum().item()
                gt_ensemble_acc[member] += gt_correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        self.llm_label_accuracy = gt_ensemble_acc / total
        
        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc