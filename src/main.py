import argparse
import json
import numpy as np
import os
import shutil
from copy import deepcopy
from tqdm import tqdm
import time

import numpy as np
from gym.spaces import Dict, Box

from Agent import Agent
from Allocator import *
from Auction import Auction
from Bidder import * 
from plot import *


def parse_kwargs(kwargs):
    parsed = ','.join([f'{key}={value}' for key, value in kwargs.items()])
    return ',' + parsed if parsed else ''

def draw_features(rng, num_runs, feature_dim, agent_configs):
    run2item_features = {}
    run2item_values = {}

    for run in range(num_runs):
        temp = []
        for k in range(agent_config['allocator']['kwargs']['num_items']):
            feature = rng.normal(0.0, 1.0, size=feature_dim)
            temp.append(feature)
        run2item_features[run] = np.stack(temp)

        run2item_values[run] = np.ones((agent_config['allocator']['kwargs']['num_items'],))

    return run2item_features, run2item_values

def set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim):
    if CTR_mode=='bilinear':
        CTR_param = rng.normal(0.0, 1.0, size=(context_dim, feature_dim))
    elif CTR_mode=='MLP':
        ################CHANGED####################
        d = context_dim + feature_dim
        w1 = rng.normal(0.0, 1.0, size=(d, d))
        ################CHANGED##################
        b1 = rng.normal(0.0, 1.0, size=(1, d))
        w2 = rng.normal(0.0, 1.0, size=(d, 1))
        b2 = rng.normal(0.0, 1.0, size=(1, 1))
        CTR_param = (w1, b1, w2, b2)
    
    if winrate_mode=='simulation':
        raise NotImplementedError
    elif winrate_mode=='logistic':
        winrate_param = rng.normal(0.0, 1.0, size=(context_dim+1,))
    elif winrate_mode=='MLP':
        d = context_dim + 1
        w1 = rng.normal(0.0, 1.0, size=(d, d))
        ###################CHANGED#################3
        b1 = rng.normal(0.0, 1.0, size=(1, d))
        w2 = rng.normal(0.0, 1.0, size=(d, 1))
        b2 = rng.normal(0.0, 1.0, size=(1, 1))
        winrate_param = (w1, b1, w2, b2)

    return CTR_param, winrate_param

if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()

    with open('config/training_config.json') as f:
        training_config = json.load(f)
    
    with open(args.config) as f:
        agent_config = json.load(f)

    # Set up Random Generator
    rng = np.random.default_rng(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    # training loop config
    num_runs = training_config['num_runs']
    num_iter  = training_config['num_iter']
    record_interval = training_config['record_interval']
    update_interval = training_config['update_interval']
    random_bidding = training_config['random_bidding']

    # context, item feature config
    context_dim = training_config['context_dim']
    feature_dim = training_config['feature_dim']
    context_dist = training_config['context_distribution']

    # CTR, winrate model
    CTR_mode = training_config['CTR_mode']
    winrate_mode = training_config['winrate_mode']

    # allocator, bidder type
    allocator_type = agent_config['allocator']['type']
    bidder_type = agent_config['bidder']['type']

    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda
    print("running in {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse configuration file
    output_dir = agent_config['output_dir']
    output_dir = output_dir + time.strftime('%y%m%d-%H%M%S')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.copy(args.config, os.path.join(output_dir, 'agent_config.json'))
    shutil.copy('config/training_config.json', os.path.join(output_dir, 'training_config.json'))

    # memory recording statistics
    reward = np.zeros((num_runs, num_iter))
    regret = np.zeros((num_runs, num_iter))
    win = np.empty((num_runs, num_iter), dtype=bool)
    optimal_reward = np.zeros((num_runs, num_iter))
    optimal_selection = np.empty((num_runs, num_iter), dtype=np.bool)
    estimated_CTR = np.zeros((num_runs, num_iter))

    true_CTR_buffer = np.zeros((record_interval))
    optimistic_CTR_buffer = np.zeros((record_interval))
    bidding_error_buffer = np.zeros((record_interval))

    num_records = int(num_iter / record_interval)
    CTR_RMSE = np.zeros((num_runs, num_records))
    CTR_bias = np.zeros((num_runs, num_records))
    optimism_ratio = np.zeros((num_runs, num_records))
    
    run2bidding_error = {}
    run2uncertainty = {}

    run2item_features, run2item_values = draw_features(rng, num_runs, feature_dim, agent_config)

    for run in range(num_runs):
        item_features = run2item_features[run]
        item_values = run2item_values[run]
        allocator = eval(f"{allocator_type}(rng=rng, item_features=item_features{parse_kwargs(agent_config['allocator']['kwargs'])})")
        bidder = eval(f"{bidder_type}(rng=rng{parse_kwargs(agent_config['bidder']['kwargs'])})")
        
        CTR_param, winrate_param = set_model_params(rng, CTR_mode, winrate_mode, context_dim, feature_dim)
        CTR_model = CTR(CTR_mode, context_dim, item_features, CTR_param)
        winrate_model = Winrate(winrate_mode, context_dim, winrate_param)
        agent = Agent(rng, agent_config['name'], item_features, item_values, allocator, bidder, context_dim, update_interval, random_bidding)
        auction = Auction(rng, agent, CTR_model, winrate_model, item_features, item_values, context_dim, context_dist)

        contexts = []
        items = []
        biddings = []
        outcomes = []

        bidding_error = []
        uncertainty = []

        context = auction.reset()
        contexts.append(context)
        for i in tqdm(range(num_iter), desc=f'run {run}'):
            ###################CHANGED#######################
            # value = agent.item_values[item]
            if isinstance(bidder, OracleBidder):
                item, estimated_CTR = agent.select_item(context)
                value = agent.item_values[item]
                b_grid = np.linspace(0.1*value, 1.5*value, 200)
                # b_grid = np.linspace(min(budget, 0.1*value), min(budget, 1.5*value), 200)
                prob_win = auction.compute_winrate(context, b_grid)
                item, bid, estimated_CTR, optimistic_CTR = agent.bid(context, value, prob_win, b_grid)
            else:
                    ############CHANGED#################
                    ############ variable names have underbar_ added
                item_, bid_, estimated_CTR_, optimistic_CTR_ = agent.bid(context)
                value_ = agent.item_values[item_]
            action = {'item' : item_, 'bid' : np.array([bid_])}
            context_, reward_, _, _, info_ = auction.step(action)
            
            items.append(item_)
            biddings.append(bid_)
            outcomes.append(info_['outcome'])

            reward[run,i] = reward_
            regret[run,i] = info_['regret']
            estimated_CTR[run,i] = estimated_CTR_
            win[run,i] = info_['win']
            optimal_selection[run,i] = info_['optimal_selection']
            optimal_reward[run,i] = info_['optimal_reward']
            optimistic_CTR_buffer[i%record_interval] = optimistic_CTR_
            true_CTR_buffer[i%record_interval] = info_['true_CTR']
            bidding_error_buffer[i%record_interval] = info_['bidding_error']


            if (i+1)%agent.update_interval==0:
                agent.update(np.array(contexts), np.array(items, dtype=int), np.array(biddings), win[run,:i+1],
                             np.array(outcomes, dtype=bool), estimated_CTR[run,:i+1], reward[run,:i+1])
 
            if (i+1)%record_interval==0:
                ind = int((i+1)/record_interval)-1
                ##############CHANGED################
                CTR_RMSE[run,ind] = np.sqrt(np.sum((estimated_CTR[run,i+1-record_interval:i+1]-true_CTR_buffer)**2)/record_interval) # i+1-record_interval
                ###############CHANGED#############
                CTR_bias[run,ind] = np.mean(estimated_CTR[run,i+1-record_interval:i+1]/(true_CTR_buffer+1e-6)) # i+1
                optimism_ratio[run,ind] = np.mean(optimistic_CTR_buffer/(estimated_CTR[run,i+1-record_interval:i+1]+1e-6)) # i+1
                bidding_error.append(bidding_error_buffer)
                bidding_error_buffer = np.zeros((record_interval))
                ###########CHANGED##############
                uncertainty.append(agent.allocator.get_uncertainty())

            contexts.append(context)
        run2bidding_error[run] = bidding_error
        run2uncertainty[run] = uncertainty
    

    experiment_id = f"CTR:{CTR_mode}_WR:{winrate_mode}_{random_bidding.replace(' ','-')}_alloc:{allocator_type}_bid:{bidder_type}"

    reward = average(reward, record_interval)
    regret = average(regret, record_interval)
    optimal_selection = average(optimal_selection.astype(float), record_interval)
    prob_win = average(win, record_interval)

    cumulative_reward = np.cumsum(reward, axis=1)
    stepwise_regret = regret
    regret = np.cumsum(stepwise_regret, axis=1)

    reward_df = numpy2df(reward, 'Reward')
    ###########CHANGED###############3
    output_file = os.path.join(experiment_id, 'reward.csv')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reward_df.to_csv(output_file, index=False)
    # reward_df.to_csv(experiment_id + '/reward.csv', index=False)
    plot_measure(reward_df, 'Reward', record_interval, experiment_id)

    cumulative_reward_df = numpy2df(cumulative_reward, 'Cumulative Reward')
    plot_measure(cumulative_reward_df, 'Cumulative Reward', record_interval, experiment_id)

    stepwise_regret_df = numpy2df(stepwise_regret, 'Stepwise Regret')
    plot_measure(stepwise_regret_df, 'Stepwise Regret', record_interval, experiment_id)

    regret_df = numpy2df(regret, 'Regret')
        ###########CHANGED###############3
    output_file = os.path.join(experiment_id, 'regret.csv')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    regret_df.to_csv(output_file, index=False)
    # regret_df.to_csv(experiment_id + '/regret.csv', index=False)
    plot_measure(reward_df, 'Regret', record_interval, experiment_id)

    optimal_selection_df = numpy2df(optimal_selection, 'Optimal Selection Rate')
            ###########CHANGED###############3
    output_file = os.path.join(experiment_id, 'optimal_selection_rate.csv')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    optimal_selection_df.to_csv(output_file, index=False)
    # optimal_selection_df.to_csv(experiment_id + '/optimal_selection_rate.csv', index=False)
    plot_measure(optimal_selection_df, 'Optimal Selection Rate', record_interval, experiment_id)

    prob_win_df = numpy2df(prob_win, 'Probability of Winning')
                ###########CHANGED###############3
    output_file = os.path.join(experiment_id, 'prob_win.csv')
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    optimal_selection_df.to_csv(output_file, index=False)
    # prob_win_df.to_csv(experiment_id + '/prob_win.csv', index=False)
    plot_measure(prob_win_df, 'Probability of Winning', record_interval, experiment_id)

    CTR_RMSE_df = numpy2df(CTR_RMSE, 'CTR RMSE')
    plot_measure(CTR_RMSE_df, 'CTR RMSE', record_interval, experiment_id)
    CTR_bias_df = numpy2df(CTR_bias, 'CTR Bias')
    plot_measure(CTR_bias_df, 'CTR Bias', record_interval, experiment_id)

    optimism_ratio_df = numpy2df(optimism_ratio, 'Optimistic/Expected CTR Ratio')
    plot_measure(optimism_ratio_df, 'Optimistic/Expected CTR Ratio', record_interval, experiment_id)
    
    bidding_error_df = list_of_numpy2df(run2bidding_error, 'Bidding Error')
    boxplot(bidding_error_df, 'Bidding Error', record_interval, experiment_id)

    uncertainty_df = list_of_numpy2df(run2uncertainty, 'Uncertainty in Parameters')
    boxplot(uncertainty_df, 'Uncertainty in Parameters', record_interval, experiment_id)