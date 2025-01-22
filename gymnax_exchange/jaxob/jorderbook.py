from functools import partial
import importlib
from os import remove
from readline import remove_history_item
from typing import Dict, NamedTuple, Optional, Tuple
from unicodedata import bidirectional
import gymnax_exchange.jaxob.JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import Configuration
job = importlib.reload(job)
import jax
from jax import numpy as jnp
from jax import lax
import chex



class LobState(NamedTuple):
    asks: jnp.ndarray
    bids: jnp.ndarray
    trades: jnp.ndarray
    key : chex.PRNGKey



class OrderBook():
    def __init__(
            self: 'OrderBook',
            cfg: Optional[Configuration] = None,
        ) -> None:
        self.cfg = cfg if cfg is not None else Configuration()

    @jax.jit
    def init(self: 'OrderBook') -> LobState:
        asks = (jnp.ones((self.cfg.nOrders, 6)) * -1).astype(jnp.int32)
        bids = (jnp.ones((self.cfg.nOrders, 6)) * -1).astype(jnp.int32)
        trades = (jnp.ones((self.cfg.nTrades, 8)) * -1).astype(jnp.int32)
        key=jax.random.PRNGKey(self.cfg.seed)
        return LobState(asks, bids, trades,key)

    @jax.jit
    def reset(
            self: 'OrderBook',
            l2_book: Optional[jnp.ndarray] = None,
            time: Optional[jnp.ndarray] = None,
        ) -> LobState:
        """"""
        state = self.init()
        if l2_book is not None:
            if time is None:
                time=jnp.array([0,0])
            msgs = job.init_msgs_from_l2(self.cfg,l2_book,time=time)
            state = self.process_orders_array(state, msgs)
        return state

    def process_order(
            self: 'OrderBook',
            state: LobState,
            quote: Dict,
            from_data: bool = False,
            verbose: bool = False
        ) -> LobState:
        '''Wrapper function for the object class that takes a Dict Object as the quote,
         ensures the order is conserved and turns the values into a jnp array which is passed to the JNP ProcessOrder function'''
        #Type, Side,quant,price
        inttype = 5
        intside = -1
        if quote['side'] == 'bid':
            intside = 1 

        if quote['type'] == 'limit':
            inttype = 1
        elif quote['type'] == 'cancel':
            inttype = 2
        elif quote['type'] == 'delete':
            inttype = 2
        elif quote['type'] == 'market':
            inttype = 1
            intside = intside * -1

        #TODO: Config so that a type 4 market order is seen as a near touch order. 

        msg = jnp.array([
            inttype,
            intside,
            quote['quantity'],
            quote['price'],
            quote['trade_id'],
            quote['order_id'],
            int(quote['timestamp'].split('.')[0]),
            int(quote['timestamp'].split('.')[1])
        ], dtype=jnp.int32)
        (*ordersides,key)=state
        key,split_key =jax.random.split(key)
        ordersides, _ = job.cond_type_side(self.cfg,ordersides,(split_key,msg))
        return LobState(*ordersides,key)

    @jax.jit
    def process_order_array(
            self: 'OrderBook',
            state: LobState,
            quote: jax.Array,
            from_data: bool = False,
            verbose: bool = False,
        ) -> LobState:
        '''Same as process_order but quote is an array.'''
        (*ordersides,key)=state
        key,split_key =jax.random.split(key)
        ordersides, _ = job.cond_type_side(self.cfg,ordersides, (split_key,quote))
        return LobState(*ordersides,key)

    @jax.jit
    def process_orders_array(
        self: 'OrderBook',
        state: LobState,
        msgs: jax.Array,
    ) -> LobState:
        '''Wrapper function for the object class that takes a JNP Array of messages (Shape=Nx8), and applies them, in sequence, to the orderbook'''
        (*ordersides,key)=state
        key,split_key =jax.random.split(key)
        return LobState(*job.scan_through_entire_array(self.cfg,split_key,msgs, tuple(ordersides)),key)

    @partial(jax.jit, static_argnums=(3,))
    def process_orders_array_l2(
            self: 'OrderBook',
            state: LobState,
            msgs: jax.Array,
            n_levels: int
        ) -> Tuple[LobState, jax.Array]:
        (*ordersides,key)=state
        key,split_key =jax.random.split(key)
        all_asks, all_bids, trades = job.scan_through_entire_array_save_states(self.cfg,split_key,msgs, tuple(ordersides), msgs.shape[0])
        state = LobState(all_asks[-1], all_bids[-1], trades,key)
        # calculate l2 states
        l2_states = jax.vmap(job.get_L2_state, (0, 0, None,None), 0)(all_asks, all_bids, n_levels,self.cfg)
        return state, l2_states

    @partial(jax.jit, static_argnums=(2,4))
    def get_volume_at_price(
            self: 'OrderBook',
            state: LobState,
            side: int,
            price: int,
            init_only: bool = False
        ) -> int:

        if side == 0:
            side_array = state.asks
        elif side == 1:
            side_array = state.bids
        else:
            raise ValueError('Side must be 0 or 1')

        if init_only:
            return job.get_init_volume_at_price(side_array, price,self.cfg)
        else:
            return job.get_volume_at_price(side_array, price)

    @jax.jit
    def get_best_price(
            self: 'OrderBook',
            state: LobState,
            side: int
        ) -> int:
        return jax.lax.cond(
            side == 1,
            lambda s: self.get_best_bid(s),
            lambda s: self.get_best_ask(s),
            state
        )
    
    @jax.jit
    def get_best_bid(
            self: 'OrderBook',
            state: LobState
        ) -> int:
        return job.get_best_bid(self.cfg,state.bids)

    @jax.jit
    def get_best_ask(
            self: 'OrderBook',
            state: LobState
        ) -> int:
        return job.get_best_ask(self.cfg,state.asks)

    @jax.jit
    def get_best_bid_and_ask_inclQuants(
            self: 'OrderBook',
            state: LobState
        ) -> Tuple[jax.Array, jax.Array]:
        return job.get_best_bid_and_ask_inclQuants(self.cfg,state.asks, state.bids)

    @partial(jax.jit, static_argnums=(2,))
    def get_L2_state(
            self: 'OrderBook',
            state: LobState,
            n_levels: int
        ) -> jax.Array:
        return job.get_L2_state(state.asks, state.bids, n_levels,self.cfg)
    
    @jax.jit
    def get_side_ids(
            self: 'OrderBook',
            state: LobState,
            side: int
        ) -> jax.Array:
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            state.asks, state.bids
        )
        return job.get_order_ids(side_array)

    @jax.jit
    def get_order(
            self: 'OrderBook',
            state: LobState,
            side: int,
            order_id: int,
            price: Optional[int] = None,
        ) -> jax.Array:
        ''' '''
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            state.asks, state.bids
        )
        if price is not None:
            return job.get_order_by_id_and_price(side_array, order_id, price)
        else:
            return job.get_order_by_id(side_array, order_id)
        
    @jax.jit
    def get_order_at_time(
            self: 'OrderBook',
            state: LobState,
            side: int,
            time_s: int,
            time_ns: int,
        ) -> jax.Array:
        ''' '''
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            state.asks, state.bids
        )
        return job.get_order_by_time(side_array, time_s, time_ns)

        
    @jax.jit
    def get_next_executable_order(
            self: 'OrderBook',
            state: LobState,
            side: int
        ):
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            state.asks, state.bids
        )
        return job.get_next_executable_order(self.cfg,side, side_array)
    
    #Flatten and Unflatten functions so that methods can be appropriately jitted. 
    def _tree_flatten(self: 'OrderBook'):
        children = ()  # arrays / dynamic values
        aux_data = {'cfg': self.cfg,}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(
    OrderBook,
    OrderBook._tree_flatten,
    OrderBook._tree_unflatten
)


if __name__ == "__main__":
    print("""Testing functionality of the jax OrderBook object in jorderbook.py 
                acts as an object wrapper around the JaxOrderBookArrays functions""")
    
    import typing

    ob=OrderBook()
    
    dict_quote={'type':'limit','side':'bid','quantity':99,'price':346000,'trade_id':8888,'order_id':8888,'timestamp':'3400.005000000'}

    l2init=jnp.array([354200,452,350100,89,361200,100,344000,400,362900,100,343100,100,364000,400,338700,100,371900,1100,337100,1000,372200,100,336400,1000,372300,200,336000,300,372800,1000,333600,1000,374600,1000,332500,100,376700,100,331600,100])
    state=ob.reset(l2init)
    print(state)
    print("Message dict quote processing",ob.process_order(state,dict_quote))
    msg_array=jnp.array([1,1,99,346000,8888,8888,3400,5000000])
    msgs_array=jnp.array([[1,1,99,346000,8888,8888,3400,5000000],
                          [1,-1,2,346000,8777,8777,3401,5060000]])

    print("Message array processing",ob.process_order_array(state,msg_array))

    print("Process array of messages and return N_levels of L2 state",ob.process_orders_array_l2(state,msgs_array,10))

    print("Volume at price test:",ob.get_volume_at_price(state,1,346000,True))

    print("Next executable order test:", ob.get_next_executable_order(state,1))

    print("Best bid price:", ob.get_best_price(state,1))

    print("Best ask price:", ob.get_best_price(state,0))

    print("Best bid and ask prices + quants:", ob.get_best_bid_and_ask_inclQuants(state))

