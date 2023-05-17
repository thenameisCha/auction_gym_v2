import numpy as np

from Allocator import *
from Bidder import *

class Agent:
    ''' An agent representing an advertiser '''

    def __init__(self, rng, name, item_features, item_values, allocator, bidder, context_dim, update_interval, random_bidding):
        self.rng = rng
        self.name = name
        self.items = item_features
        self.item_values = item_values

        self.num_items = item_features.shape[0]
        self.feature_dim = item_features.shape[1]
        self.context_dim = context_dim

        self.allocator = allocator
        self.bidder = bidder

        split = random_bidding.split()
        self.random_bidding_mode = split[0]    # uniform or gaussian noise
        if self.random_bidding_mode!='None':
            self.init_num_random_bidding = int(split[1])
            self.decay_factor = float(split[2])

        self.use_optimistic_value = True

        self.clock = 0
        self.update_interval = update_interval
    
    def should_explore(self):
        if isinstance(self.bidder, OracleBidder):
            return False
        if (self.allocator.mode=='TS' or self.allocator.mode=='UCB') and self.use_optimistic_value:
            return self.clock%self.update_interval < \
            self.init_num_random_bidding/np.power(self.decay_factor, int(self.clock/self.update_interval))/4
        else:
            return self.clock%self.update_interval < \
                self.init_num_random_bidding/np.power(self.decay_factor, int(self.clock/self.update_interval))

    def select_item(self, context):
        # Estimate CTR for all items
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
            estim_CTRs = self.allocator.estimate_CTR(context, UCB=True)
        elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
            estim_CTRs = self.allocator.estimate_CTR(context, TS=True)
        else:
            estim_CTRs = self.allocator.estimate_CTR(context)
        # Compute value if clicked
        estim_values = estim_CTRs * self.item_values
        best_item = np.argmax(estim_values)
        if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='Epsilon-greedy':
            if self.rng.uniform(0,1)<self.allocator.eps:
                best_item = self.rng.choice(self.num_items, 1).item()

        return best_item, estim_CTRs[best_item]

    def bid(self, context, value=None, prob_win=None, b_grid=None):
        self.clock += 1
        item, estimated_CTR = self.select_item(context)
        optimistic_CTR = estimated_CTR
        value = self.item_values[item]

        if isinstance(self.bidder, OracleBidder):
            bid = self.bidder.bid(value, estimated_CTR, prob_win, b_grid)
        elif not isinstance(self.allocator, OracleAllocator) and self.should_explore():
            if self.random_bidding_mode=='Uniform':
                bid = self.rng.uniform(0, value*1.5)
            elif self.random_bidding_mode=='Overbidding-uniform':
                bid = self.rng.uniform(value*1.0, value*1.5)
            elif self.random_bidding_mode=='Gaussian':
                bid = self.bidder.bid(value, context, estimated_CTR)
                bid += value * self.rng.normal(0, 0.5)
                bid = np.maximum(bid, 0)
        else:
            if not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='UCB':
                mean_CTR = self.allocator.estimate_CTR(context, UCB=False)
                estimated_CTR = mean_CTR[item]
                if self.use_optimistic_value:
                    bid = self.bidder.bid(value, context, optimistic_CTR)
                else:
                    bid = self.bidder.bid(value, context, estimated_CTR)
            elif not isinstance(self.allocator, OracleAllocator) and self.allocator.mode=='TS':
                mean_CTR = self.allocator.estimate_CTR(context, TS=False)
                estimated_CTR = mean_CTR[item]
                if self.use_optimistic_value:
                    bid = self.bidder.bid(value, context, optimistic_CTR)
                else:
                    bid = self.bidder.bid(value, context, estimated_CTR)
            else:
                bid = self.bidder.bid(value, context, estimated_CTR)
        return item, bid, estimated_CTR, optimistic_CTR

    def update(self, context, item, bid, won, outcome, estimated_CTR, reward):
        # Update response model with data from winning bids
        self.allocator.update(context[won], item[won], outcome[won], self.name)

        # Update bidding model with all data
        self.bidder.update(context, bid, outcome, self.name)