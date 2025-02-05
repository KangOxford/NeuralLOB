import os
import sys
import time
import csv
import datetime
from os import listdir
from os.path import isfile,join

import jax
import flax
import numpy as np
import jax.numpy as jnp

sys.path.append('/homes/80/kang/AlphaTrade')
from gymnax_exchange.jaxrl.ppoS5ExecCont import ActorCriticS5
from gymnax_exchange.jaxrl.ppoS5ExecCont import StackedEncoderModel, ssm_size, n_layers
from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from AlphaTrade.gymnax_exchange.jaxen.old.exec_env_old import *

ppo_config = {    
    "ENV_NAME": "alphatradeExec-v0",
    "DEBUG": True,
    "TASKSIDE":'sell',

    "REWARD_LAMBDA":0,
    # "ACTION_TYPE":"pure",
    "ACTION_TYPE":"delta",
    "TASK_SIZE": 100, #500,
    
    "WINDOW_INDEX": -1,
    # "ATFOLDER": "/homes/80/kang/AlphaTrade/testing", # testing one Month data
    "ATFOLDER": "/homes/80/kang/AlphaTrade/testing_oneDay",
    "RESULTS_FILE":"/homes/80/kang/AlphaTrade/results_file_"+f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
    
    "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_11-10_03-13/", # N.O. 81, delta quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-19_06-27/", # N.O. 23, pure quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_00-33/", # N.O. 11, pure quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-15_10-03/", # N.O. 11, pure quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-14_10-16/", # N.O. 10, pure quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-11_04-22/", # N.O. 3, pure quant
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/ckpt/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-07_09-09/",
    # "CHECKPOINT_DIR":"/homes/80/kang/AlphaTrade/checkpoints_10-06_12-57/",
}

dir = ppo_config['CHECKPOINT_DIR']
csv_dir = ppo_config['CHECKPOINT_DIR']+"csv/"


def make_evaluation(network_config):    
    network,trainstate_params,checkpoint,env,env_params,key_step = network_config
    def step(rng, obs, done, hstate,state):
        
        ac_in = (obs[np.newaxis,np.newaxis, :], jnp.array([done])[np.newaxis, :])
        assert len(ac_in[0].shape) == 3, f"{ac_in[0].shape}"
        assert len(ac_in[1].shape) == 2, f"{ac_in[1].shape}"
        hstate, pi, value = network.apply(trainstate_params, hstate, ac_in) 
        raw_action = pi.sample(seed=rng)
        # action = raw_action.round().astype(jnp.int32)[0,0,:].clip(0, None)
        action = raw_action[0,0,:]
        obs,state,reward,done,info=env.step(key_step, state,action, env_params)
        row_data = [
            info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0,0,0],raw_action[0,0,1],
            # checkpoint, info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0,0,0],raw_action[0,0,1],
            info['done'], 
            # info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], info['drift_reward'], 
            info['quant_executed'], 
            info['task_to_execute'], info['total_revenue']
        ]
        return row_data,(obs, done, hstate,state,reward,info)
    return step


    
    
def evaluate_savefile(paramsFile,window_idx):
    # env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx)
    env= ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx,ppo_config["ACTION_TYPE"],ppo_config["TASK_SIZE"],ppo_config["REWARD_LAMBDA"])
    env_params=env.default_params
    assert env.task_size == 100 # 500
    # Automatically create the directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)
    with open(csv_dir+paramsFile.split(".")[0]+f'_wdw_idx_{window_idx}.csv', 'w', newline='') as csvfile:
        print(paramsFile)
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        # row_title = [
        #     'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','quant_executed', 'task_to_execute', 'total_revenue'
        # ]
        # csvwriter.writerow(row_title)
        # csvfile.flush() 
    
        with open(dir+paramsFile, 'rb') as f:
            trainstate_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"pramas restored")
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs,state=env.reset(key_reset,env_params)
        network = ActorCriticS5(env.action_space(env_params).shape[0], config=ppo_config)
        hstate = StackedEncoderModel.initialize_carry(1, ssm_size, n_layers)
        done = False
        
        checkpoint = paramsFile.split("_")[1].split(".")[0]
        # checkpoint = int(paramsFile.split("_")[1].split(".")[0])
        network_config = (network,trainstate_params,checkpoint,env,env_params,key_step)
        device = jax.devices()[-1]
        evaluate_jit = jax.jit(make_evaluation(network_config), device=device)    
        
        rng = jax.device_put(jax.random.PRNGKey(0), device)
        for i in range(1,10000):
            print(i)

            # start = time.time()
            rng, _rng = jax.random.split(rng)
            row_data,(obs, done, hstate,state,reward,info) = evaluate_jit(rng, obs,done,hstate,state)
            # print(f"time taken: {time.time()-start}")
            
            csvwriter.writerow(row_data)
            csvfile.flush() 
            if done:
                break
            
def twap_evaluation(paramsFile,window_idx):
    env=ExecutionEnv(ppo_config['ATFOLDER'],ppo_config["TASKSIDE"],window_idx)
    env_params=env.default_params
    assert env.task_size == 500
    # Automatically create the directory if it doesn't exist
    os.makedirs(csv_dir, exist_ok=True)    
    with open(csv_dir+paramsFile.split(".")[0]+f'_twap_{window_idx}.csv', 'w', newline='') as csvfile:
        print(paramsFile)
        csvwriter = csv.writer(csvfile)
        # Add a header row if needed
        # row_title = [
        #     'checkpiont_name','window_index', 'current_step' , 'average_price', 'delta_sum',"delta_aggressive",'delta_passive','raw_delta_aggressive','raw_delta_passive','done', 'slippage', 'price_drift', 'advantage_reward', 'drift_reward','quant_executed', 'task_to_execute', 'total_revenue'
        # ]
        # csvwriter.writerow(row_title)
        # csvfile.flush() 
    
        rng = jax.random.PRNGKey(0)
        rng, key_reset, key_step = jax.random.split(rng, 3)
        obs,state=env.reset(key_reset,env_params)
        done = False
        for i in range(1,10000):
            print(i)
            raw_action = jnp.array([0, 0])
            # action = raw_action.round().astype(jnp.int32)[0,0,:].clip(0, None)
            action = raw_action
            obs,state,reward,done,info=env.step(key_step, state,action, env_params)
            row_data = [
                info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0],raw_action[1],
                # paramsFile.split("_")[1].split(".")[0], info['window_index'], info['current_step'], info['average_price'], action.sum(), action[0], action[1],raw_action[0],raw_action[1],
                info['done'],
                # info['done'], info['slippage'], info['price_drift'], info['advantage_reward'], info['drift_reward'], 
                info['quant_executed'], 
                info['task_to_execute'], info['total_revenue']
            ]
            csvwriter.writerow(row_data)
            csvfile.flush() 
            if done:
                break
    
import re
import argparse

def PPO_main(idx=-1):
    start_time =time.time()
    # for window_idx in range(int(1e5)):
    # for window_idx in range(260):
    for window_idx in range(13):
        # try:
        start=time.time()
        print(f">>> window_idx: {window_idx}")
        def extract_number_from_filename(filename):
            match = re.search(r'_(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0  # default if no number is found

        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
        onlyfiles = sorted(onlyfiles, key=extract_number_from_filename)
        paramsFile = onlyfiles[idx]
        evaluate_savefile(paramsFile,window_idx)
        print(f"Time for evaluation: \n",time.time()-start)
        # except:
        #     print(f"End of the window index is {window_idx}")
        #     break
    print(f"Total time for evaluation ppo : \n",time.time()-start_time)
    
def TWAP_main(idx=-1):
    start_time =time.time()
    # for window_idx in range(int(1e5)):
    # for window_idx in range(260):
    for window_idx in range(13):
        # try:
        start=time.time()
        print(f">>> window_idx: {window_idx}")        
        def extract_number_from_filename(filename):
            match = re.search(r'_(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0  # default if no number is found

        onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
        onlyfiles = sorted(onlyfiles, key=extract_number_from_filename)
        paramsFile = onlyfiles[idx]
        twap_evaluation(paramsFile,window_idx)
        print(f"Time for evaluation: \n",time.time()-start)
        # except:
        #     print(f"End of the window index is {window_idx}")
        #     break
    print(f"Total time for evaluation twap: \n",time.time()-start_time)


# ===========================================================================================
# your_checkpoint_dir (a directory)
#     => csv (a directory)
#         params_file_electric-waterfall-87_10-07_12-52_twap_0.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_1.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_2.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_3.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_4.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_5.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_6.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_7.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_8.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_9.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_10.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_11.csv
#         params_file_electric-waterfall-87_10-07_12-52_twap_12.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_0.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_1.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_2.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_3.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_4.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_5.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_6.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_7.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_8.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_9.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_10.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_11.csv
#         params_file_electric-waterfall-87_10-07_12-52_wdw_idx_12.csv
#     => params_file_electric-waterfall-87_10-07_12-52 (your trained network params, a file)
#     => params_file_electric-waterfall-87_10-07_13-52 (second network params during training)
# ===========================================================================================
'''
generated by func <PPO_main> : params_file_electric-waterfall-87_10-07_12-52_twap_0.csv
generated by func <TWAP_main>: params_file_electric-waterfall-87_10-07_12-52_wdw_idx_0.csv
args.idx to be -1            : select the last params file, in our situation, it is 
                               params_file_electric-waterfall-87_10-07_13-52
'''
def comparison_main(idx=-1):
    PPO_main(idx)
    TWAP_main(idx)
    
    
def plotting(number_string):
    '''
    number_string is the checkpoints number in string format
    it would be used in:
    dir = f"/homes/80/kang/AlphaTrade/checkpoints_10-15_10-03/csv/checkpoint_{number_string}_wdw_idx_{idx}.csv"
    which means the csv result file which contains results evaluated from the idx-th data_window
    and the number_string-th checkpoint means the num_string-th saved ppo params during the training
    
    dir = f"/homes/80/kang/AlphaTrade/testing_oneDay/prices/results_file_numpy_{idx}.npy"
    means the data path of the prices file of the idx-th data_window, which is used to test the performance of the ppo agent
    these files are generated by the base_env.py:
        # ---------------------------------------
        # for j in range(13):
        #     data = Cubes_withOB[j][0][:,:,3]
        #     dir = f"/homes/80/kang/AlphaTrade/testing_oneDay/prices/results_file_numpy_{j}.npy"
        #     np.save(dir, data)
        # ---------------------------------------
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_subfigure(ax1, x_values, y1_values, y2_values, df_sample):
        ax1.plot(x_values, y1_values, linestyle='-', label='Quantity Executed', color='b')
        ax1.set_xlabel('Current Step')
        ax1.set_ylabel('Quantity Executed', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        start_point = np.argmax(y1_values > 0)
        end_point = np.argmax(y1_values >= 499)
        if end_point == 0:
            end_point = len(y1_values) - 1

        ax1.axhspan(0, 500, xmin=start_point/len(y1_values), xmax=end_point/len(y1_values), facecolor='grey', alpha=0.5)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(x_values, y2_values, linestyle='-', label='Prices', color='r')
        ax2.set_ylabel('Prices', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')

        last_avg_price = int(df_sample['average_price'].iloc[-1]*100)
        ax2.axhline(y=last_avg_price, color='r', linestyle='--', label=f'Avg Price: {last_avg_price}')
        ax2.legend(loc='lower right')

    # Create a 4x4 grid
    plt.figure(figsize=(40, 40))


    import numpy as np
    import pandas as pd
    for idx in range(13):
        print(idx)
        ax1 = plt.subplot(4, 4, idx+1)

        dir = f"/homes/80/kang/AlphaTrade/testing_oneDay/prices/results_file_numpy_{idx}.npy"
        loaded_data = np.load(dir)
        first_row =loaded_data[:,0]
        prices = first_row[first_row>0]

        dir = f"/homes/80/kang/AlphaTrade/checkpoints_10-15_10-03/csv/checkpoint_{number_string}_wdw_idx_{idx}.csv"
        # dir = f"/homes/80/kang/AlphaTrade/checkpoints_10-14_10-16/csv/checkpoint_34126000_wdw_idx_{idx}.csv"
        # dir = f"/homes/80/kang/AlphaTrade/checkpoints_10-11_04-22/csv/checkpoint_34853000_wdw_idx_{idx}.csv"
        df = pd.read_csv(dir)
        df_sample = df.copy()
        len_prices = len(prices)
        len_df = len(df_sample['current_step'])
        if len_df > len_prices:
            extended_prices = np.pad(prices, (0, len_df - len_prices), 'constant', constant_values=500)
            x_values = df_sample['current_step']
            y1_values = df_sample['quant_executed']
            y2_values = extended_prices
        else:
            extended_df = np.pad(df_sample['quant_executed'], (0, len_prices - len_df), 'constant', constant_values=500)
            x_values = np.arange(1, len_prices + 1)
            y1_values = extended_df
            y2_values = prices
            
        plot_subfigure(ax1, x_values, y1_values, y2_values, df_sample)


    plt.show()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify index of the file to evaluate.')
    parser.add_argument('--idx', metavar='idx', type=int, default=-1, help='Index of the file to evaluate.')
    args = parser.parse_args()
    
    comparison_main(args.idx)

    # /bin/python3 /homes/80/kang/AlphaTrade/gymnax_exchange/test_scripts/evaluation.py 2
    # /bin/python3 /homes/80/kang/AlphaTrade/gymnax_exchange/test_scripts/evaluation.py -1

