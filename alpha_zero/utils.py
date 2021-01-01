import numpy as np
from alpha_zero.mcts import get_best_action, get_best_q, mcts, Node, get_ns

def eval_winrate(totest, bench, env, model, n_games=100):
    wins = 0
    draws = 0
    rsum = 0
    for igame in range(n_games):
        done, reward = False, 0
        obs = env.reset()
        curr_policy = totest if igame<=n_games//2 else bench
        rew2count = 1 if igame<=n_games//2 else -1
        while not done:
            action = curr_policy.act(obs, model.available_actions(obs))
            obs, r, done, _ = env.step(action)
            rsum += r
            curr_policy = totest if curr_policy==bench else bench
            wins += 1 if r==rew2count else 0
        draws += 1 if r==0 else 0
    winrate = wins/n_games
    drawrate = draws/n_games
    return winrate, drawrate, (1-winrate-drawrate)


def play_mcts_against_human(env, model, policy, rollout_policy, human='first', discount_factor=0.99, n_simulations=100):
    obs = env.reset()
    env.render()
    done = False
    turn = 0
    human = 0 if human=='first' else 1 
    while not done:
        print("-"*20)
        if human == turn%2:
            besta = int(input("your play? "))
            print("your action was", besta)
            obs, r, done, _ = env.step(besta)
            env.render()
        else:
            nodes = {obs:Node()}
            for i in range(n_simulations):
                mcts(nodes, obs, policy, model, rollout_policy, cpucb=10, discount_factor=discount_factor)
            besta = get_best_action(nodes, obs, model)
            best_q =get_best_q(nodes, obs, model)
            print(get_ns(nodes, obs, model))
            print(besta, best_q)
            obs, r, done, _ = env.step(besta)
            env.render()
        turn += 1
        print("\n"*2)
    print("final reward=", r)


def play_mcts_against_itself(env, model, policy, rollout_policy, discount_factor=0.99, n_simulations=100):
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        nodes = {obs:Node()}
        for i in range(n_simulations):
            mcts(nodes, obs, policy, model, rollout_policy, cpucb=10, discount_factor=discount_factor)
        besta = get_best_action(nodes, obs, model)
        best_q =get_best_q(nodes, obs, model)
        print(get_ns(nodes, obs, model))
        print(besta, best_q)
        obs, r, done, _ = env.step(besta)
        env.render()
    print("final reward=", r)

def save(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    torch.save(model.state_dict(), PATH)
def load(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    model.load_state_dict(torch.load(PATH))