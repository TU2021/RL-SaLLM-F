gpt_query_env_prompts_button_press = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to press a button.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object position that the robot needs to touch, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the position of the target button, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """

gpt_query_env_drawer_open = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to open the drawer.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object drawer handle position that the robot needs to touch, TCP needs to pull the handle out of the drawer, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the position of the target drawer, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """


gpt_query_env_drawer_close = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to open the drawer.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object drawer handle position that the robot needs to touch, TCP needs to pull the handle out of the drawer, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the position of the target drawer, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """


gpt_query_env_door_open = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to open the door with a revolving joint.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object door revolving joint position that the robot needs to touch, TCP needs to rotate the revolving joint and pull the door open, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the target final position of the door, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """

gpt_query_env_window_open = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to push and open the window.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object window handle position that the robot needs to touch, TCP needs to push the handle and open the window, and obj's coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the target final position of the window, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """

gpt_query_env_door_unlock = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to unlock the door by rotating the lock counter-clockwise.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object door rlock counter-clockwise position that the robot needs to touch, TCP needs to rotate the lock counter-clockwiseand unlock the door, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the target final position of the lock counter-clockwise, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
"""

gpt_query_env_coffee_push = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to push a mug under a coffee machine.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object mug position that the robot needs to touch, TCP needs to touch the mug cube first, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the target final position under a coffee machine, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
"""

gpt_query_env_handle_pull = """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to pull a handle up.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot actuator, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “obj” represents the object handle position that the robot needs to touch, TCP needs to touch and hold the bottom of the handle first, and its coordinates should change, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (3) “target” represents the target final position of the handle, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """

gpt_query_env_reach =  """Suppose you are a good robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory.
    The goal is to control the Tool Center Point (TCP) of the robot to reach a goal.
    The following are two trajectories, which contain {} steps, where:
    (1) “tcp” represents the end position of the robot gripper (TCP), which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];
    (2) “target” represents the target goal position, which is expressed in three-dimensional Cartesian coordinates in the range of [0,1];

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.
    """

gpt_query_env_UMaze = """
    Suppose you are an excellent robot trajectory evaluator. Now you need to evaluate the quality of the agent's trajectory.
    
    The map of the environment is as follows:
        [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
    where 1 indicates that there is a wall in this cell, 0 indicates that this cell is free for the agent and goal.

    The goal is to control the agent to reach the goal while preventing it from hitting the walls.
    The following are two trajectories, which contain {} steps, where:
    (1) "position" represents the two-dimensional coordinates (x,y) of the agent, the unit is m.
    (2) "target" represents the target goal position, the unit is m.

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.

    """


gpt_query_env_Open = """
    Suppose you are an excellent robot trajectory evaluator. Now you need to evaluate the quality of the agent's trajectory.
    
    The map of the environment is as follows:
        [[1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]]
    where 1 indicates that there is a wall in this cell, 0 indicates that this cell is free for the agent and goal.

    The goal is to control the agent to reach the goal while preventing it from hitting the walls.
    The following are two trajectories, which contain {} steps, where:
    (1) "position" represents the two-dimensional coordinates (x,y) of the agent, the unit is m.
    (2) "target" represents the target goal position, the unit is m.

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.

    """


gpt_query_env_Medium = """
    Suppose you are an excellent robot trajectory evaluator. Now you need to evaluate the quality of the agent's trajectory.
    
    The map of the environment is as follows:
        [[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]
    where 1 indicates that there is a wall in this cell, 0 indicates that this cell is free for the agent and goal.

    The goal is to control the agent to reach the goal while preventing it from hitting the walls.
    The following are two trajectories, which contain {} steps, where:
    (1) "position" represents the two-dimensional coordinates (x,y) of the agent, the unit is m.
    (2) "target" represents the target goal position, the unit is m.

    {}

    Please answer the following two questions step by step: 
    1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? 
    Reply your analysis.

    2. Which trajectory you think do better with achieving the goal?
    Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significantly difference.

    """


gpt_query_env_prompts = {
     "metaworld_button-press-v2": gpt_query_env_prompts_button_press,
     "metaworld_drawer-open-v2": gpt_query_env_drawer_open,
     "metaworld_drawer-close-v2": gpt_query_env_drawer_close,
     "metaworld_door-open-v2": gpt_query_env_door_open,
     "metaworld_window-open-v2": gpt_query_env_window_open,
     "metaworld_door-unlock-v2": gpt_query_env_door_unlock,
     "metaworld_handle-pull-v2":gpt_query_env_handle_pull,
     "metaworld_reach-v2":gpt_query_env_reach,
     "PointMaze_UMazeDense-v3":gpt_query_env_UMaze,
     "PointMaze_OpenDense-v3":gpt_query_env_Open,
     "PointMaze_MediumDense-v3":gpt_query_env_Medium,
    }



trajgen_template_for_robot = """
    Based on your analysis, Can you generate a new trajectory based on the initial state of that good trajectory that you think can better achieve the goal?
    Replay only the generate better trajectory. 
    The generated trajectory should meet the following characteristics:
    1. The movement of TCP should be smooth and touch obj as quickly as possible, then the change should conform to the laws of physics, change smoothly, avoid sudden changes in coordinates and finally the obj should reach the target;
    2. TCP should first move to the position of obj, that is, the coordinates of TCP and obj should be at similar values, and then push obj to move;
    3. Output a trajectory that conforms to the input trajectory format, the step size should be {} and the trajectory should be started with
    {}
    .
"""


trajgen_template_for_maze = """
    Based on your analysis, Can you generate a new trajectory based on the initial state of that good trajectory that you think can better achieve the goal?
    Replay only the generate better trajectory. 
    The generated trajectory should meet the following characteristics:
    1. The movement of agent should be smooth and get to the target position as quickly as possible, then the change should conform to the laws of physics, change smoothly, avoid sudden changes in coordinates and finally the obj should reach the target;
    2. During the entire agent movement process, attention should be paid to the walls in the map to avoid collisions with them;
    3. Output a trajectory that conforms to the input trajectory format, including position and target, the step size should be {} and the trajectory should be started with
    {}
    .
"""


def trajgen_template(env):
    if "metaworld" in env:
        return trajgen_template_for_robot
    elif "PointMaze" in env:
        return trajgen_template_for_maze