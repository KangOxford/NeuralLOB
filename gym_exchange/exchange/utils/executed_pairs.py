import numpy as np
class ExecutedPairs():
    def __init__(self):
        self.market_pairs = []
        self.agent_pairs  = []
        
    def step(self, trades, kind):
        if len(trades) == 0: pass
        else: # len(trades) == 1 or 3
            batch = self.trades2pairs(trades)
            self.update(batch, kind)
        
    def trades2pairs(self, trades):
        pairs = [[trade['price'], trade['quantity']] for trade in trades]
        formatted_pairs = np.array(pairs).T
        '''pairs   | 3000 | 3001
           quantity|   1  |  2  '''
        return formatted_pairs 
    
    def update(self, pairs, kind):
        if kind == "market": self.market_pairs.append(pairs)
        elif kind=="agent" : self.agent_pairs.append(pairs)
        else: raise NotImplementedError
        
        
""" trades format
transaction_record = {
        'timestamp': self.time,
        'price': traded_price,
        'quantity': traded_quantity,
        'time': self.time
        }
if side == 'bid':
    transaction_record['party1'] = [counter_party, 'bid', head_order.order_id, new_book_quantity]
    transaction_record['party2'] = [quote['trade_id'], 'ask', None, None]
else:
    transaction_record['party1'] = [counter_party, 'ask', head_order.order_id, new_book_quantity]
    transaction_record['party2'] = [quote['trade_id'], 'bid', None, None]
"""

''' pairs format
price:    array([[ 1. ,  1. ,  1. ,  1.1,  0.9],
quantity:        [ 2. , 23. ,  3. , 21. ,  3. ]])
'''

'''trade format
{'timestamp': '34201.40462348', 'price': 31180000, 'quantity': 1, 'time': '34201.40462348', 
'party1': [3032093, 'ask', 3032093, None], 
'party2': [15750757, 'bid', None, None]}
'''

if __name__ == "__main__":
    pass