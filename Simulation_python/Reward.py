#!/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getReward3(iifds, actions, obs, global_reward, all_opp):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][5] * 3 + obs[car_id][7]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][1] * 3 + obs[car_id][3]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id] != []:
            has_enemy = True
        else:
            has_enemy = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0:  # 这里包含了死亡的情况
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards


def getReward2(iifds, actions, obs, global_reward, all_opp):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:追击, 2:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][5] * 3 + obs[car_id][7]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][1] * 3 + obs[car_id][3]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id] != []:
            has_enemy = True
        else:
            has_enemy = False

        if has_enemy:
            # 存在敌军时策略
            if actions[car_id] == 1:  # 正确执行追击
                rewards[car_id] += correct_action
            else:  # 错误动作
                rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0:  # 这里包含了死亡的情况
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 2:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards


def getReward1(iifds, actions, obs, global_reward, all_opp, all_nei_c2e, obs_dim):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][int(2*obs_dim/4+1)] * 3 + obs[car_id][int(3*obs_dim/4+1)]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][int(0*obs_dim/4+1)] * 3 + obs[car_id][int(1*obs_dim/4+1)]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id]:
            has_enemy = True
        else:
            has_enemy = False

        if all_nei_c2e[car_id]:
            has_ally = True
        else:
            has_ally = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0 and has_ally:  # 这里包含了死亡的情况
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards

def getReward2(iifds, actions, obs, global_reward, all_opp, all_nei_c2e, obs_dim):
    num_agent = int(iifds.numberofcar / 2)
    rewards = np.zeros(num_agent)
    correct_action = 1
    wrong_action = -1
    # actions: 0:搜索, 1:逃跑, 2:追击, 3:支援
    for car_id in range(num_agent):
        situation_enemy = obs[car_id][int(2 * obs_dim / 4 + 1)] * 3 + obs[car_id][
            int(3 * obs_dim / 4 + 1)]  # 敌方态势得分（有利态势追击，反之逃跑）
        situation_nei = obs[car_id][int(0 * obs_dim / 4 + 1)] * 3 + obs[car_id][
            int(1 * obs_dim / 4 + 1)]  # 友方态势得分（相对朝向友方时支援，反之搜索）

        if all_opp[car_id+num_agent]:
            has_enemy = True
        else:
            has_enemy = False

        if all_nei_c2e[car_id+num_agent]:
            has_ally = True
        else:
            has_ally = False

        if has_enemy:
            # 存在敌军时策略
            if situation_enemy >= 0:
                if actions[car_id] == 2:  # 正确执行追击
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 1:  # 正确执行逃跑
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        else:
            # 没有敌军时策略
            if situation_nei >= 0 and has_ally:  # 这里包含了死亡的情况
                if actions[car_id] == 3:  # 正确执行支援
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
            else:
                if actions[car_id] == 0:  # 正确执行搜索
                    rewards[car_id] += correct_action
                else:  # 错误动作
                    rewards[car_id] += wrong_action
        rewards[car_id] += global_reward

    return rewards

