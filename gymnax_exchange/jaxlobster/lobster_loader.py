"""
#TODO: Update to fit reality. 

University of Oxford
Corresponding Author: 
Kang Li     (kang.li@keble.ox.ac.uk)
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)
Peer Nagy   (peer.nagy@reuben.ox.ac.uk)
V1.0

Module Description
This module loads data from load_LOBSTER, initializes orders data in cubes from
 messages for downstream tasks, and creates auxiliary arrays with 
 essential information for these tasks.

Key Components 
The return of the funcion is:
    Cubes_withOB: a list of (cube, OB), where cube is of of three dimension
                  0-aixs is index of data windows 
                  1-axis is index of steps inside the data window
                  2-axis is index of lines of data inside the steps
    max_steps_in_episode_arr: horizon(max steps) of one data window
    taskSize_array: the amount of share for n%(default as 1) of 
                    the traded volume in the window

Functionality Overview
load_files:             loads the csvs as pandas arrays
preProcessingMassegeOB: adjust message_day data and orderbook_day data. 
sliceWithoutOverlap:    split the message of one day into multiple cubes.
Cubes_withOB_padding:   pad the cubes to have the shape shape
                        to be stored in single array.
"""
from os import listdir
from os.path import isfile, join
import warnings

import itertools
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np

from jax import numpy as jnp
import jax
from jax import lax


class LoadLOBSTER():
    """
    Class which completes all of the loading from the lobster data
    set files. 

    ...

    Attributes
    ----------
    atpath : str
        Path to the "AlphaTrade" repository folder
    messagePath : str
        Path to folder containing all message data
    orderbookPath : str
        Path to folder containing all orderbook state data
    window_type : str
        "fixed_time" or "fixed_steps" defines whether episode windows
        are defined by time (e.g. 30mins) or n_steps (e.g. 150)
    window_length : int
        Length of an episode window. In seconds or steps (see above)
    n_messages : int
        number of messages to process from data per step

    Methods
    -------
    run_loading():
        Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        Additionally returns initial state data for each window, and 
        the lengths (horizons) of each window. 
    """
    def __init__(self,
                 alphatradepath,
                 n_Levels=10,
                 type_="fixed_time",
                 window_length=1800,
                 n_msg_per_step=100):
        self.atpath=alphatradepath
        self.messagePath = alphatradepath+"/data/Flow_"+str(n_Levels)+"/"
        self.orderbookPath = alphatradepath+"/data/Book_"+str(n_Levels)+"/"
        self.window_type=type_
        self.window_length=window_length
        self.n_messages=n_msg_per_step


    def run_loading(self):
        """Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        
            Parameters:
                NA   
            Returns:
                loaded_msg_windows (Array): messages sliced into fixed-
                                            size windows.
                                            Dimensions: 
                                            (Window, Step, Message, Features)
        """
        message_days, orderbook_days = self._load_files()
        pairs = [self._pre_process_msg_ob(msg,ob) 
                 for msg,ob 
                 in zip(message_days,orderbook_days)]
        message_days, orderbook_days = zip(*pairs)
        slicedCubes_withOB_list = [self._slice_day_no_overlap(msg_day,ob_day) 
                                   for msg_day,ob_day 
                                   in zip(message_days,orderbook_days)]
        cubes_withOB = list(itertools.chain \
                            .from_iterable(slicedCubes_withOB_list))
        max_steps_in_windows_arr = jnp.array([m.shape[0] 
                                              for m,o 
                                              in cubes_withOB],jnp.int32)
        cubes_withOB=self._pad_window_cubes(cubes_withOB)
        loaded_msg_windows,loaded_book_windows=map(jnp.array,
                                                    zip(*cubes_withOB))
        n_windows=len(loaded_book_windows)
        return (loaded_msg_windows,
                loaded_book_windows,
                max_steps_in_windows_arr,
                n_windows)

    def _pad_window_cubes(self,cubes_withOB):
        #Get length of longest window
        max_win = max(w.shape[0] for w, o in cubes_withOB)
        new_cubes_withOB = []
        for cube, OB in cubes_withOB:
            cube = self._pad_cube(cube, max_win)
            new_cubes_withOB.append((cube, OB))
        return new_cubes_withOB
    
    def _pad_cube(self, cube, target_shape):
        """Given a 'cube' of data, representing one episode window, pad
        it with extra entries of 0 to reach a target number of steps.
        """
        # Calculate the amount of padding required
        padding = [(0, target_shape - cube.shape[0]), (0, 0), (0, 0)]
        padded_cube = np.pad(cube, padding, mode='constant', constant_values=0)
        return padded_cube

    def _slice_day_no_overlap(self, message_day, orderbook_day):
        """ Given a day of message and orderbook data, slice it into
        'cubes' of episode windows. 
        """
        sliced_parts, init_OBs = self._split_day_to_windows(message_day, orderbook_day)
        slicedCubes = [self._slice_to_cube(slice_) for slice_ in sliced_parts]
        slicedCubes_withOB = zip(slicedCubes, init_OBs)
        return slicedCubes_withOB

    def _load_files(self):
        """Loads the csvs as pandas arrays. Files are seperated by days
        Could potentially be optimised to work around pandas, very slow.         
        """
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(self.messagePath), readFromPath(self.orderbookPath)
        dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
        messageCSVs = [pd.read_csv(self.messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
        orderbookCSVs = [pd.read_csv(self.orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
        return messageCSVs, orderbookCSVs
    
    def _pre_process_msg_ob(self,message_day,orderbook_day):
        """Adjust message_day data and orderbook_day data. 
        Splits time into two fields, drops unused message_day types,
        transforms executions into limit orders and delete into cancel
        orders, and adds the traderID field. 
        """
        #split the time into two integer fields.
        message_day[6] = message_day[0].apply(lambda x: int(x))
        message_day[7] = ((message_day[0] - message_day[6]) * int(1e9)).astype(int)
        message_day.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns']
        #Drop all message_days of type 5,6,7 (Hidden orders, Auction, Trading Halt)
        message_day = message_day[message_day.type.isin([1,2,3,4])]
        valid_index = message_day.index.to_numpy()
        message_day.reset_index(inplace=True,drop=True)
        # Turn executions into limit orders on the opposite book side
        message_day.loc[message_day['type'] == 4, 'direction'] *= -1
        message_day.loc[message_day['type'] == 4, 'type'] = 1
        #Turn delete into cancel orders
        message_day.loc[message_day['type'] == 3, 'type'] = 2
        #Add trader_id field (copy of order_id)
        warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        message_day['trader_id'] = message_day['order_id']
        orderbook_day.iloc[valid_index,:].reset_index(inplace=True, drop=True)
        return message_day,orderbook_day
    
    def _daily_slice_indeces(self,type,start, end, interval):
        """Returns a list of times of indices at which to cut the daily
        message data into data windows.
            Parameters:
                type (str): "fixed_steps" or "fixed_time" mode
                start (int): start time of the day or index of
                                first message to consider
                end (int): end time or last index to consider.
                            
                interval (int): length of an episode window in
                                terms of time (s) or number of 
                                steps.
            Returns:
                    indices (List): Either times or indices at which
                    to slice the data array. 
        """
        if type == "fixed_steps":
            end_index = ((end-start)
                         // self.n_messages*self.n_messages+start+1)
            indices = list(range(start, end_index, self.n_messages*interval))
        elif type == "fixed_time":
            indices = list(range(start, end+1, interval))
        else: raise NotImplementedError('Use either "fixed_time" or' 
                                        + ' "fixed_steps"')
        if len(indices)<2:
            raise ValueError("Not enough range to get a slice")
        return indices

    def _split_day_to_windows(self,message_day,orderbook_day):
        """Splits a day of messages into given windows.
        The windows are either defined by a fixed time interval or a 
        fixed number of steps, whereby each step is a fixed number of 
        messages.
        For each window, the initial orderbook state is taken from the
        orderbook data. 
            Parameters:
                message (Array): Array of all messages in a day
                orderbook (Array): All order book states in a day.
                                    1st dim of equal size as message. 
           Returns:
                sliced_parts (List): List of arrays each repr. a window.  
                init_OBs (List): List of arrays repr. init. orderbook
                                    data for each window. 
        """
        d_end = (message_day['time_s'].max()+1 
                 if self.window_type=="fixed_time"  
                 else message_day.shape[0])
        d_start = (message_day['time_s'].min() 
                 if self.window_type=="fixed_time"  
                 else 0)
        indices=self._daily_slice_indeces(self.window_type,
                                         d_start,
                                         d_end,
                                         self.window_length)
        sliced_parts = []
        init_OBs = []
        for i in range(len(indices) - 1):
            start_index = indices[i]
            end_index = indices[i + 1]
            if self.window_type == "fixed_steps":
                sliced_part = message_day[(message_day.index > start_index) &
                                             (message_day.index <= end_index)]
            elif self.window_type == "fixed_time":
                index_s, index_e = message_day[(message_day['time'] >= start_index) &
                                            (message_day['time'] < end_index)].index[[0, -1]].tolist()
                index_e = ((index_e // self.n_messages - 1) * self.n_messages
                            + index_s % self.n_messages)
                assert ((index_e - index_s) 
                        % self.n_messages == 0), 'wrong code 31'
                sliced_part = message_day.loc[np.arange(index_s, index_e)]
            sliced_parts.append(sliced_part)
            init_OBs.append(orderbook_day.iloc[start_index,:])
        
        if self.window_type == "fixed_steps":
            print(indices)
            print(len(sliced_parts))
            assert len(sliced_parts) == len(indices)-1, 'wrong code 33'
            for part in sliced_parts:
                assert part.shape[0] % self.n_messages == 0, 'wrong code 34'
        elif self.window_type == "fixed_time":
            for part in sliced_parts:
                assert part.shape[0] % self.n_messages == 0, 'wrong code 34'
        return sliced_parts, init_OBs
    
    def _slice_to_cube(self,sliced):
        """Turn a 2D pandas table of messages into a 3D numpy array
        whereby the additional dimension is due to the message
        stream being split into fixed-size blocks"""
        columns = ['type','direction','qty','price',
                   'trader_id','order_id','time_s','time_ns']
        cube = sliced[columns].to_numpy()
        cube = cube.reshape((-1, self.n_messages, 8))
        return cube


class LoadLOBSTER_resample():
    """
    Class which completes all of the loading from the lobster data
    set files as a single array of all messages of interest.
    
    Assumes that the split into padded chunks happens 'live'.
    Provides jittable methods to make this happen.

    ...

    Attributes
    ----------
    atpath : str
        Path to the "AlphaTrade" repository folder
    messagePath : str
        Path to folder containing all message data
    orderbookPath : str
        Path to folder containing all orderbook state data
    window_type : str
        "fixed_time" or "fixed_steps" defines whether episode windows
        are defined by time (e.g. 30mins) or n_steps (e.g. 150)
    window_length : int
        Length of an episode window. In seconds or steps (see above)
    window_resolution : int 
        Places at which a window may start. Every minute, 
            or N-thousand step based on window_type. 
    n_messages : int
        number of messages to process from data per step (omits option
        to consider a fixed time per step)
    
    

    Methods
    -------
    run_loading():
        Returns jax.numpy arrays with messages sliced into fixed-size
        windows. Dimensions: (Window, Step, Message, Features)
        Additionally returns initial state data for each window, and 
        the lengths (horizons) of each window. 
    """
    def __init__(self,
                 alphatradepath,
                 n_Levels=10,
                 type_="fixed_time",
                 window_length=1800,
                 window_resolution=60,
                 n_msg_per_step=100):
        self.atpath=alphatradepath
        self.messagePath = alphatradepath+"/data/Flow_"+str(n_Levels)+"/"
        self.orderbookPath = alphatradepath+"/data/Book_"+str(n_Levels)+"/"
        self.window_type=type_
        self.window_length=window_length
        self.window_resolution=window_resolution
        self.n_messages=n_msg_per_step
        self.index_offest=0



    def run_loading(self):
        """Returns jax.numpy array with all messages aligned in series. 
        
            Parameters:
                NA   
            Returns:
                loaded_msgs (Array): messages 
                                            Dimensions: 
                                            (Messages, Features)
                max_window_size (Int)
        """
        message_days, orderbook_days = self._load_files()
        pairs = [self._pre_process_msg_ob(msg,ob) 
                 for msg,ob 
                 in zip(message_days,orderbook_days)]
        message_days, orderbook_days = zip(*pairs)
        

        #Get the 'window' indices of starts & ends for each of the days.
        #Get the lengths of all possible windows given those starts.
        pairs = [self._get_inits_day(msg_day,ob_day) 
                                   for msg_day,ob_day 
                                   in zip(message_days,orderbook_days)]
        msgs,starts,ends,obs = zip(*pairs)
        
        #Concatenate the data from all the days.
        msgs=jnp.concatenate(msgs,0)
        
        starts=jnp.concatenate(starts,0)
        ends=jnp.concatenate(ends,0)
        obs=jnp.concatenate(obs,0)

        max_msgs_in_windows_arr=ends-starts
        (msgs,
         max_msgs_in_windows_arr)=self._pad_last_ep(msgs,
                                                       max_msgs_in_windows_arr)
        return msgs,starts,ends,obs,max_msgs_in_windows_arr
    
    def _pad_last_ep(self,messages,max_msgs_in_windows_arr):
        length_last_ep=max_msgs_in_windows_arr[-1]
        new_length=(length_last_ep//self.n_messages+1)*self.n_messages
        pad=jnp.zeros((new_length-length_last_ep,messages.shape[1]),dtype=jnp.int32)
        last_time=jnp.array([messages[-1,-2:][0]+1,0])
        pad=pad.at[:,-2:].set(last_time)
        messages=jnp.concatenate((messages,pad))
        max_msgs_in_windows_arr=max_msgs_in_windows_arr.at[-1].set(new_length)
        return messages,max_msgs_in_windows_arr


    def _load_files(self):
        """Loads the csvs as pandas arrays. Files are seperated by days
        Could potentially be optimised to work around pandas, very slow.         
        """
        readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
        messageFiles, orderbookFiles = readFromPath(self.messagePath), readFromPath(self.orderbookPath)
        dtype = {0: float,1: int, 2: int, 3: int, 4: int, 5: int}
        messageCSVs = [pd.read_csv(self.messagePath + file, usecols=range(6), dtype=dtype, header=None) for file in messageFiles if file[-3:] == "csv"]
        orderbookCSVs = [pd.read_csv(self.orderbookPath + file, header=None) for file in orderbookFiles if file[-3:] == "csv"]
        return messageCSVs, orderbookCSVs
    
    def _pre_process_msg_ob(self,message_day,orderbook_day):
        """Adjust message_day data and orderbook_day data. 
        Splits time into two fields, drops unused message_day types,
        transforms executions into limit orders and delete into cancel
        orders, and adds the traderID field. 
        """
        #split the time into two integer fields.
        message_day[6] = message_day[0].apply(lambda x: int(x))
        message_day[7] = ((message_day[0] - message_day[6]) * int(1e9)).astype(int)
        message_day.columns = ['time','type','order_id','qty','price','direction','time_s','time_ns']
        #Drop all message_days of type 5,6,7 (Hidden orders, Auction, Trading Halt)
        message_day = message_day[message_day.type.isin([1,2,3,4])]
        valid_index = message_day.index.to_numpy()
        message_day.reset_index(inplace=True,drop=True)
        # Turn executions into limit orders on the opposite book side
        message_day.loc[message_day['type'] == 4, 'direction'] *= -1
        message_day.loc[message_day['type'] == 4, 'type'] = 1
        #Turn delete into cancel orders
        message_day.loc[message_day['type'] == 3, 'type'] = 2
        #Add trader_id field (copy of order_id)
        warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
        message_day['trader_id'] = message_day['order_id']
        orderbook_day.iloc[valid_index,:].reset_index(inplace=True, drop=True)
        return message_day,orderbook_day
    
    def _daily_slice_indeces(self,type,start, end, interval):
        """Returns a list of times of indices at which an episode
        window may start.
            Parameters:
                type (str): "fixed_steps" or "fixed_time" mode
                start (int): start time of the day or index of
                                first message to consider
                end (int): end time or last index to consider.
                            
                interval (int): length between starts in
                                terms of time (s) or number of 
                                steps.
            Returns:
                    indices (List): Either times or indices at which
                    to slice the data array. 
        """
        if type == "fixed_steps":
            end_index = ((end-start)
                         // self.n_messages*self.n_messages+start+1)
            indices = list(range(start, end_index, self.n_messages*interval))
        elif type == "fixed_time":
            indices = list(range(start, end+1, interval))
        else: raise NotImplementedError('Use either "fixed_time" or' 
                                        + ' "fixed_steps"')
        if len(indices)<2:
            raise ValueError("Not enough range to get a slice")
        return indices

    def _get_inits_day(self,message_day,orderbook_day):
        """Obtains the starting indeces for each of the possible
        message windows and a list of the initial book states at
        these times. Doesn't return any message data due to this
        not being sliced at this time (expected on reset).

            Parameters:
                message (Array): Array of all messages in a day
                orderbook (Array): All order book states in a day.
                                    1st dim of equal size as message. 
           Returns:
                indices (Array): Array of indices of starting points.
                init_OBs (List): List of arrays repr. init. orderbook
                                    data for each starting point.
        """
        d_end = (message_day['time_s'].max()+1-self.window_length+self.window_resolution
                 if self.window_type=="fixed_time"  
                 else message_day.shape[0]-
                    self.window_length*self.n_messages)
        d_start = (message_day['time_s'].min() 
                 if self.window_type=="fixed_time"  
                 else 0)
        #Note indices may be either time or index. Confusing. 
        indices=self._daily_slice_indeces(self.window_type,
                                         d_start,
                                         d_end,
                                         self.window_resolution)
        index_s = []
        index_e = []
        if self.window_type == "fixed_steps":
                index_s=np.array(indices)
                index_e=np.array(indices)+np.ones_like(index_s)*self.n_messages*self.window_length
                max_msgs=self.n_messages*self.window_length

        elif self.window_type == "fixed_time":
            for i in range(len(indices) - 1):
                window_start = indices[i]
                window_end = indices[i] + self.window_length
                # Filter the messages for the current window
                filtered = message_day[
                    (message_day['time'] >= window_start) &
                    (message_day['time'] < window_end)
                ]
                # Debug: print out information about the candidate window
                #print(f"Window {i}: start time {window_start}, end time {window_end}, "
                #    f"found {len(filtered)} rows in the message data.")
                if not filtered.empty:
                    # Get the first and last row indices of this window
                    (i_s, i_e) = filtered.index[[0, -1]]
                    #print(f"  Window {i} indices: first index = {i_s}, last index = {i_e}")
                    index_s.append(i_s)
                    index_e.append(i_e)
                else:
                    # If no data is found, print a warning (seems to happen quite often for smaller window sizes)
                    print(f"  Warning: Window {i} has no data!")

        init_OBs=jnp.array(orderbook_day.iloc[jnp.array(index_s),:])
        index_s=jnp.array(index_s)+jnp.ones_like(jnp.array(index_s))*self.index_offest
        index_e=jnp.array(index_e)+jnp.ones_like(jnp.array(index_e))*self.index_offest
        self.index_offest=self.index_offest+message_day.shape[0]
        columns = ['type','direction','qty','price',
                   'trader_id','order_id','time_s','time_ns']
        message_day=message_day[columns].to_numpy()
        return jnp.array(message_day),index_s,index_e,init_OBs
    



if __name__ == "__main__":
    #Load data from 50 Levels, fixing each episode to 150 steps
    #containing 100 messages each. 
    loader=LoadLOBSTER_resample("/AlphaTrade/training_oneDay",10,"fixed_time",window_length=1800,n_msg_per_step=100,window_resolution=60)
    msgs,starts,ends,obs,max_msgs=loader.run_loading()
    print(msgs.shape)
    print(starts.shape)
    print(ends.shape)
    print(obs.shape)
    print(max_msgs)

    print(starts[720:722])

    print(msgs[starts[720:722]])


    print(msgs[-100:])


    """
    #Load data from 50 Levels, fixing each episode to 30 minutes
    #(1800 seconds) containing a varied number of steps of 100
    # messages each.
    loader=LoadLOBSTER_resample("./AlphaTrade",10,"fixed_steps",window_length=300,n_msg_per_step=100,window_resolution=5)
    msgs,starts,ends,obs,max_msgs=loader.run_loading()
    print(msgs.shape)
    print(starts.shape)
    print(ends.shape)
    print(obs.shape)
    print(max_msgs)


    loader=LoadLOBSTER("./AlphaTrade",10,"fixed_time",window_length=1800,n_msg_per_step=100)
    msgs,obs,max_steps,n_windows=loader.run_loading()
    print(msgs.shape)
    print(msgs[0,0,0,:])
    """
