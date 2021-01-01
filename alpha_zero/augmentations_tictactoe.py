import torch

def flip_sign(tensor_obs, idx):
    signs = [1, -1]
    return tensor_obs*signs[idx]
def flip_board(tensor_obs, idx):
    ops = [[], [0], [1], [0, 1]]
    return torch.flip(tensor_obs, ops[idx])
def rotate_board(tensor_obs, idx):
    none = lambda x:x
    rot  = lambda x:torch.rot90(x, 1, [0, 1])
    ops = [none, rot]
    return ops[idx](tensor_obs)
def symetric_add2rbuff(rbuff, list_with_boards, rew_sign, r):
    added_boards = set()
    for tensor_obs, monte_probs, rew_sign in list_with_boards:
        for sign in range(2):
            for flip in range(4):
                for rotate in range(2):
                    side_len = 3#int(math.sqrt(tensor_obs.shape[0]*tensor_obs.shape[1]))
                    # print(side_len)
                    board_obs = tensor_obs.view(1, -1)
                    board_obs, player = board_obs[:, :-1], board_obs[:, -1].unsqueeze(-1)
                    board_obs = board_obs.reshape(side_len, side_len)
                    sign_boardt = flip_sign(board_obs, sign)
                    player = flip_sign(player, sign)
                    flip_boardt = flip_board(sign_boardt, flip)
                    board = rotate_board(flip_boardt, rotate)
                    board = board.reshape(1, -1)
                    board = torch.cat((board, player), dim=-1)
                    board = board.view(1, 1, -1)


                    board_monte = monte_probs.reshape(side_len, side_len)
                    # sign_montet = flip_sign(board_monte, sign)
                    flip_montet = flip_board(board_monte, flip)
                    monte = rotate_board(flip_montet, rotate)
                    monte = monte.reshape(1, 1, -1)

                    if board not in added_boards:
                        added_boards.add(board)
                        raw_reward = torch.tensor([[[r]]]).float()
                        flip_rew_sign = 1 if sign==0 else -1
                        reward_now = raw_reward*flip_rew_sign*rew_sign
                        rbuff.add(board, monte, reward_now)