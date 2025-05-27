import numpy as np
import re
import json

def obs_process(env_name, obs, process_type=0):
    # obs: array, shape=(39,)
    if process_type == 0:
        return obs
    
    elif 'metaworld' in env_name:
        assert obs.size == 39
        env_name = env_name.replace('metaworld_','')
        # For specific MetaWorld environments, customize observation processing
        if env_name == "reach-v2":
            if process_type == 1:
                process_obs = np.concatenate([obs[0:3], obs[-3:]])
            if process_type == 2:
                process_obs = np.concatenate([obs[0:3], obs[18:21], obs[-3:]])
            else: 
                process_obs = obs
        else:
            if process_type == 1:
                process_obs = np.concatenate([obs[0:7], obs[-3:]])
            if process_type == 2:
                process_obs = np.concatenate([obs[0:7], obs[18:25], obs[-3:]])
            else: 
                process_obs = obs
        return process_obs
    
    else:
        return obs

def obs_process_for_batch(env_name, obs, process_type=0):
    # obs: array, shape=(n,39)
    obs_repeat = obs
    data_trimmed = obs_repeat[:-1, :]
    # Duplicate the first row and insert it at the beginning
    first_row = data_trimmed[0:1, :]  # Keep the two-dimensional structure
    obs_repeat = np.vstack([first_row, data_trimmed])

    process_obs = np.hstack([obs[:, 0:7], obs_repeat[:, 0:4], obs[:, 18:25-4], obs[:, -3:]])

    return process_obs
    

def get_process_obs_dim(env_name, obs, process_type):
    return obs_process(env_name, obs, process_type).shape

def traj_pair_process(env_name, sa_t_1, sa_t_2, r_t_1, r_t_2, labels):
    if 'metaworld' in env_name:
        env_name = env_name.replace('metaworld_','')
        if env_name == "reach-v2":
            tcp_1 = sa_t_1[:,0:3].squeeze()
            tcp_2 = sa_t_2[:,0:3].squeeze()
            target_1 = sa_t_1[0,-3:].squeeze()
            target_2 = sa_t_2[0,-3:].squeeze()
            total_reward_1  = np.sum(r_t_1, axis=0).squeeze()
            total_reward_2  = np.sum(r_t_2, axis=0).squeeze()
            labels = labels.squeeze()

            # Store data for a single batch as a dictionary
            data_to_save = {
                "tcp_1": tcp_1.tolist(),          # Convert to list format for saving
                "target_1": target_1.tolist(),
                "total_reward_1": total_reward_1.tolist(),  # Save total reward

                "tcp_2": tcp_2.tolist(),
                "target_2": target_2.tolist(),
                "total_reward_2": total_reward_2.tolist(),

                "labels": labels.tolist()
            }

            traj_1 = {
                "tcp": tcp_1.tolist(),          # Convert to list format for saving
                "target": target_1.tolist(),
            }

            traj_2 = {
                "tcp": tcp_2.tolist(),          # Convert to list format for saving
                "target": target_2.tolist(),
            }

            return data_to_save, traj_1, traj_2  # Return saved data dictionary    
        else:
            tcp_1 = sa_t_1[:,0:3].squeeze()
            tcp_open_1 = sa_t_1[:,3].squeeze()
            tcp_2 = sa_t_2[:,0:3].squeeze()
            tcp_open_2 = sa_t_2[:,3].squeeze()
            obj_1 = sa_t_1[:,4:7].squeeze()
            obj_2 = sa_t_2[:,4:7].squeeze()
            target_1 = sa_t_1[0,-3:].squeeze()
            target_2 = sa_t_2[0,-3:].squeeze()
            total_reward_1  = np.sum(r_t_1, axis=0).squeeze()
            total_reward_2  = np.sum(r_t_2, axis=0).squeeze()
            labels = labels.squeeze()

            # Store data for a single batch as a dictionary
            data_to_save = {
                "tcp_1": tcp_1.tolist(),          # Convert to list format for saving
                "tcp_open_1": tcp_open_1.tolist(),
                "obj_1": obj_1.tolist(),
                "target_1": target_1.tolist(),
                "total_reward_1": total_reward_1.tolist(),  # Save total reward

                "tcp_2": tcp_2.tolist(),
                "tcp_open_2": tcp_open_2.tolist(),
                "obj_2": obj_2.tolist(),
                "target_2": target_2.tolist(),
                "total_reward_2": total_reward_2.tolist(),

                "labels": labels.tolist()
            }

            traj_1 = {
                "tcp": tcp_1.tolist(),          # Convert to list format for saving
                "obj": obj_1.tolist(),
                "target": target_1.tolist(),
            }

            traj_2 = {
                "tcp": tcp_2.tolist(),          # Convert to list format for saving
                "obj": obj_2.tolist(),
                "target": target_2.tolist(),
            }

            return data_to_save, traj_1, traj_2  # Return saved data dictionary
        
    elif 'PointMaze' in env_name:
        pos_1 = sa_t_1[:,0:2].squeeze()
        pos_2 = sa_t_2[:,0:2].squeeze()
        v_1 = sa_t_1[:,2:4].squeeze()
        v_2 = sa_t_2[:,2:4].squeeze()
        target_1 = sa_t_1[0,-2:].squeeze()
        target_2 = sa_t_2[0,-2:].squeeze()
        total_reward_1  = np.sum(r_t_1, axis=0).squeeze()
        total_reward_2  = np.sum(r_t_2, axis=0).squeeze()
        labels = labels.squeeze()

        # Store data for a single batch as a dictionary
        data_to_save = {
            "pos_1": pos_1.tolist(),          # Convert to list format for saving
            "v_1": v_1.tolist(),
            "target_1": target_1.tolist(),
            "total_reward_1": total_reward_1.tolist(),  # Save total reward

            "pos_2": pos_2.tolist(),
            "v_2": v_2.tolist(),
            "target_2": target_2.tolist(),
            "total_reward_2": total_reward_2.tolist(),

            "labels": labels.tolist()
        }

        traj_1 = {
            "position": pos_1.tolist(),          # Convert to list format for saving
            "target": target_1.tolist(),
        }

        traj_2 = {
            "position": pos_2.tolist(),          # Convert to list format for saving
            "target": target_2.tolist(),
        }

        return data_to_save, traj_1, traj_2  # Return saved data dictionary
        
        
def format_traj_for_gpt(traj_1, traj_2):
    # Convert traj_1 and traj_2 into a formatted string
    def format_value(value):
        # If the value is a numeric type, keep 4 decimal places
        if isinstance(value, (float, int)):
            return f"{value:.4f}"
        # If the value is a list or tuple, recursively process each element
        elif isinstance(value, (list, tuple)):
            return [format_value(v) for v in value]
        # Otherwise, convert the value to a string
        else:
            return str(value)

    def format_traj(traj):
        traj_str = ""
        for key, value in traj.items():
            # Format each value
            formatted_value = format_value(value)
            # Remove unnecessary spaces and line breaks
            value_str = str(formatted_value).replace(' ', '').replace('\n', '')
            traj_str += f"{key}: {value_str}\n"
        return traj_str.strip()  # Remove trailing line breaks and spaces

    # Format traj_1 and traj_2
    traj_1_str = format_traj(traj_1)
    traj_2_str = format_traj(traj_2)

    # Combine the strings of traj_1 and traj_2
    combined_str = f"Trajectory 1:\n{traj_1_str}\n\nTrajectory 2:\n{traj_2_str}"

    return combined_str

def format_traj_begin_for_gpt(traj):
    # Convert the trajectory into a formatted string
    def format_value(value):
        # If the value is numeric, keep 4 decimal places
        if isinstance(value, (float, int)):
            return f"{value:.4f}"
        # If the value is a list or tuple, process each element recursively
        elif isinstance(value, (list, tuple)):
            return [format_value(v) for v in value]
        # Otherwise, convert the value to a string
        else:
            return str(value)

    def format_traj(traj):
        traj_str = ""
        for key, value in traj.items():
            # Format each value
            formatted_value = format_value(value)
            # Remove unnecessary spaces and line breaks
            value_str = str(formatted_value).replace(' ', '').replace('\n', '')
            traj_str += f"{key}: {value_str}\n"
        return traj_str.strip()  # Remove trailing line breaks and spaces

    # Format the trajectory
    traj_str = format_traj(traj)

    return traj_str

def extract_all_values(data_str, keyword, size):
    """Find all bracketed values after a specific keyword and parse them into a list"""
    # Locate the keyword, allowing for spaces, tabs, quotes, etc.
    pattern = fr'["\']?{keyword}["\']?\s*:\s*'

    start_idx = re.search(pattern, data_str)
    if not start_idx:
        start_idx = re.search(fr"{keyword}:\s*", data_str)
        return None  # Return None if the keyword is not found
    start_idx = start_idx.end()  # Move past the keyword

    # Find the opening bracket
    open_bracket_idx = data_str.find('[', start_idx)
    # Match the opening bracket to its corresponding closing bracket
    close_bracket_idx = find_matching_bracket(data_str, open_bracket_idx)

    if open_bracket_idx == -1 or close_bracket_idx == -1:
        return None  # Return None if no matching brackets are found

    # Extract the content within the brackets
    values_str = data_str[open_bracket_idx:close_bracket_idx+1]

    # Use regex to match all numbers
    numbers = re.findall(r"-?\d+\.\d+", values_str)

    # Group the numbers as tuples and return as a numpy array
    values = [float(num) for num in numbers]
    return np.array(values).reshape(-1, size)

def find_matching_bracket(data_str, open_idx):
    """Find the closing bracket matching the given opening bracket position"""
    stack = 0
    for idx in range(open_idx, len(data_str)):
        if data_str[idx] == '[':
            stack += 1
        elif data_str[idx] == ']':
            stack -= 1
        if stack == 0:
            return idx
    return -1  # Return -1 if no matching closing bracket is found

def extract_trajectory_from_text(env_name, trajectory_str, size_segment):
    if "metaworld" in env_name:
        tcp = 'tcp'
        obj = 'obj'
        target = 'target'
        size = 3
    elif "PointMaze" in env_name:
        tcp = 'position'
        target = 'target'
        size = 2

    try:
        # Attempt to extract values for each key
        tcp_array = extract_all_values(trajectory_str, tcp, size)
        target_array = extract_all_values(trajectory_str, target, size)
    except Exception as e:
        print("Error extract_all_values.")
        return None

    # Validate the lengths and shapes
    if tcp_array is None or target_array is None:
        print("Invalid trajectory: missing tcp or target.")
        return None

    if len(tcp_array) != size_segment:
        print(f"Invalid trajectory: tcp length is {len(tcp_array)} instead of {size_segment}.")
        if len(tcp_array) >= size_segment:
            tcp_array = tcp_array[:size_segment]
        else:
            return None

    if len(target_array) != 1:
        print(f"Invalid trajectory: target length is {len(target_array)} instead of 1.")
        return None

    if tcp_array.shape != (size_segment, size):
        print(f"Invalid trajectory: tcp shape is {tcp_array.shape} instead of ({size_segment}, {size}).")
        return None

    if target_array.shape != (1, size):
        print(f"Invalid trajectory: target shape is {target_array.shape} instead of (1, {size}).")
        return None

    return {
        tcp: tcp_array,
        target: target_array
    }

def get_response_answer(res):
    # Extract the textual reply from GPT-4
    try:
        # Extract the content
        reply_text = res.strip()

        # Use regex to extract the last number
        numbers = re.findall(r'\d+', reply_text)
        if numbers:
            final_number = int(numbers[-1])  # Take the last number as an integer
        else:
            final_number = -1  # Default value if no numbers are found
    except Exception as e:
        print(f"Error processing reply: {e}")
        final_number = -1  # Default value on error

    return final_number

def convert_trajdist_to_traj(env_name, worse_traj, new_traj_dist):
    if "metaworld" in env_name:
        """
        Convert `new_traj_dist` to match the structure of `worse_traj`
        """
        n_steps, n_dims = worse_traj.shape

        better_traj = np.copy(worse_traj)

        better_traj[:, 0:3] = new_traj_dist['tcp']

        better_traj[:, 3] = worse_traj[:, 3]

        better_traj[:, 4:7] = new_traj_dist['obj']

        better_traj[:, 14:17] = new_traj_dist['target']

        better_traj[1:, 7:10] = new_traj_dist['tcp'][:-1]
        better_traj[0, 7:10] = worse_traj[0, 7:10]

        better_traj[:, 10] = worse_traj[:, 10]

        better_traj[1:, 11:14] = new_traj_dist['obj'][:-1]
        better_traj[0, 11:14] = worse_traj[0, 11:14]

        return better_traj

    elif "PointMaze" in env_name:
        """
        Convert `new_traj_dist` to match the structure of `worse_traj`
        """
        n_steps, n_dims = worse_traj.shape

        better_traj = np.copy(worse_traj)

        better_traj[:, 0:2] = new_traj_dist['position']

        better_traj[:, -2:] = new_traj_dist['target']

        return better_traj
