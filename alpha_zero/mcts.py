def mcts(nodes, state, policy, model, rollout_policy=None, value_function=None, neg_rew=True, cpucb=15, discount_factor=1):
    if nodes[state].n == 0:
        if rollout_policy is not None:
            r = rollout_policy.rollout(state, model)
        elif value_function is not None:
            r = value_function.get(state)
        else:
            raise "neither rollout_policy nor value_function specified"
    else:
        action = ucb(nodes, state, policy, model, cpucb=cpucb, neg_rew=neg_rew)
        state2, r, done, _ = model.step(state, action)
        nodes[state].children[action] = state2
        if state2 not in nodes:
            nodes[state2] = Node(action=action, parent=state)
        if not done:
            r += discount_factor*mcts(nodes, state2, policy, model, rollout_policy, value_function, neg_rew, cpucb)
        else:
            nodes[state2].n += 1    
            nodes[state2].q += -r if neg_rew else r

    nodes[state].n += 1
    nodes[state].q += r
    return -r if neg_rew else r

def ucb(nodes, state, policy, model, cpucb=15, neg_rew=True):
    ava = model.available_actions(state)
    probs = policy.get_action_probs(state, list(range(9)))
    maxscore, a_maxscore = float('-inf'), -1
    for action in ava:
        if action in nodes[state].children:
            state_child = nodes[state].children[action]
            q = nodes[state_child].q/nodes[state_child].n * (-1 if neg_rew else 1)
            u = probs[action]/(1+nodes[state_child].n)
            score = q + cpucb*u
            if score > maxscore:
                maxscore = score
                a_maxscore = action
        else:
            return action
    return a_maxscore

class Node:
    def __init__(self, action=None, parent=None):
        self.q = 0
        self.n = 0
        self.children = {}
        self.action = action 
        self.parent = parent
    def __repr__(self):
        return "n="+str(self.n)+" q="+str(self.q)+" nchild="+str(len(self.children))+" action="+str(self.action)#+" parent="+str(self.parent)


def get_best_action(nodes, state, model):
    best_action, max_n = -1, -1
    available_actions = model.available_actions(state)
    for i, state2 in enumerate(nodes):
        if i > len(available_actions):
            break
        if i != 0 and max_n<nodes[state2].n:
          max_n = nodes[state2].n
          best_action = nodes[state2].action
    if max_n == -1:
        best_action = available_actions[0]
    return best_action

def get_best_q(nodes, state, model, neg_rew=True):
    best_value, max_n = -1, -1
    available_actions = model.available_actions(state)
    mult = -1 if neg_rew else 1
    for i, state2 in enumerate(nodes):
        if i > len(available_actions):
            break
        if i != 0 and max_n<nodes[state2].n:
          max_n = nodes[state2].n
          best_value = nodes[state2].q /max_n*mult
    if max_n == -1:
        best_value = nodes[state].q
    return best_value

def get_qs(nodes, state, model, nan_number=float('-inf')):
    qs = [nan_number for i in range(model.get_num_actions())]
    available_actions = model.available_actions(state)
    for action in available_actions:
        state_child = nodes[state].children[action]
        qs[action] = nodes[state_child].q/nodes[state_child].n
    return qs

def get_ns(nodes, state, model, nan_number=float('-inf')):
    ns = [nan_number for i in range(model.get_num_actions())]
    available_actions = model.available_actions(state)
    for action in available_actions:
        state_child = nodes[state].children[action]
        ns[action] = nodes[state_child].n
    return ns