#!/usr/bin/python

# 模拟异构群体对抗的任务，主要涉及多个无人车（包括红方、蓝方）和一个无人机在模拟环境中执行任务、攻击和逃避，并且有路径规划和奖励机制。
# 在当前细窄长墙体场景下，路径规划算法效果不是特别理想。关键函数为 iifds.detect，iifds.assign，iifds.getvNext。
# assign函数放了其中一种任务分配例子用于理解。

# 引入了许多常用的库，如 torch、numpy、matplotlib 等，用于深度学习、数值计算、图形绘制等。
import torch
import numpy as np
from All_fun import IIFDS
from Reward import getReward1
from arguments import parse_args
import random
import argparse
from battle import Battle
import pandas as pd
import os
import sys
from Network.PS_IQL_network import MultiAgentDQN

sys.path.append('../Data_csv')

seed = random.randint(1, 1000)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 将MLPActor添加到安全全局列表（为了减少运行中的Warning）
# torch.serialization.add_safe_globals([MLPActor])


def get_args():
    parser = argparse.ArgumentParser("car swarm confrontation")
    iifds = IIFDS(X_range, Y_range, obsCenter, obsR)
    # Train
    parser.add_argument("--num-RCARs", type=int, default=int(iifds.numberofcar / 2), help="number of red CARs")
    parser.add_argument("--num-BCARs", type=int, default=int(iifds.numberofcar / 2), help="number of blue CARs")
    parser.add_argument("--num-Bcars", type=int, default=1, help="number of blue cars")

    parser.add_argument("--detect-range", type=float, default=iifds.R_1, help="")

    parser.add_argument("--attack-range-B", type=float, default=iifds.threshold, help="")
    parser.add_argument("--attack-range-R", type=float, default=iifds.threshold, help="")
    parser.add_argument("--attack-angle-BR", type=float, default=iifds.hit_angle / 2, help="")

    parser.add_argument("--sensor-range-B-l", type=float, default=16, help="")
    parser.add_argument("--sensor-range-B-w", type=float, default=12, help="")
    parser.add_argument("--sensor-angle-B", type=float, default=np.pi, help="")
    # DQN参数
    parser.add_argument("--hidden-dim", type=int, default=128, help="DQN隐藏层维度")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=0.01, help="测试时使用较小的探索率")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="最小探索率")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="探索率衰减")
    parser.add_argument("--tau", type=float, default=0.01, help="目标网络软更新参数")
    parser.add_argument("--buffer-size", type=int, default=100000, help="经验回放缓冲区大小")
    parser.add_argument("--batch-size", type=int, default=256, help="批处理大小")
    parser.add_argument("--warmup-episodes", type=int, default=30, help="预热回合数")

    # 测试参数
    parser.add_argument("--num-episodes", type=int, default=100, help="测试回合数")
    parser.add_argument("--max-steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--model-path", type=str, default="saved_models/best_dqn_ps_model.pth", help="模型路径")

    args = parser.parse_args()
    return args


def test_model(args, obsCenter):
    iifds = IIFDS(X_range, Y_range, obsCenter, obsR)  # 使用 IIFDS 来处理无人车行为的细节（例如任务分配、路径规划等）。
    # iifds.clear_folder('test_results')
    arglist = parse_args()
    # 测试参数
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    blue_agents = int(iifds.numberofcar / 2)  # 蓝方智能体数量

    # ===========================
    # 初始化动力学约束以及随机参数（马子豪）
    # ===========================
    random_params = {
        # 动力学约束随机参数
        "cars_mass": np.clip(np.random.normal(2.5, 0.5, iifds.numberofcar), 2, 3),
        "cars_force": np.clip(np.random.normal(17.5, 2.5, iifds.numberofcar), 15, 20),
        "cars_power": np.clip(np.random.normal(45, 5, iifds.numberofcar), 40, 50),
        "cars_friction_coefficient": np.clip(np.random.normal(0.3, 0.02, iifds.numberofcar), 0.28, 0.32),
        "collision_coefficient": np.clip(np.random.normal(0.95, 0.05, 1), 0.9, 1).item(),
        # 感知随机参数
        "cars_position_noise": np.clip(np.random.normal(0.01, 0.002, 1), 0.008, 0.012).item(),
    }

    from physical_law import PhysicalLaw

    physical_law = PhysicalLaw(
        cars_mass=random_params["cars_mass"],  # 小车质量, 1维向量
        cars_force=random_params["cars_force"],  # 小车动力, 1维向量
        cars_power=random_params["cars_power"],  # 小车功率, 1维向量
        cars_friction_coefficient=random_params["cars_friction_coefficient"],  # 小车的摩擦系数, 1维向量
        cars_size=[[iifds.carR * 2, iifds.carR]] * iifds.numberofcar,  # 小车的长宽, 2维向量
        cars_wheel_spacing=[1] * iifds.numberofcar,  # 小车的轮间距, 1维向量
        cars_wheel_radius=[2] * iifds.numberofcar,  # 小车的轮半径, 1维向量
        obstacles_center=obsCenter[:, :2],  # 障碍物的中心点, 2维向量
        obstacles_radius=0.1,  # 障碍物的半径, 1维向量
        timestep=iifds.timeStep,  # 时间步长
        collision_coefficient=random_params["collision_coefficient"],  # 碰撞后速度的衰减系数
    )

    from perception_random import PerceptionPosition

    perception_q = PerceptionPosition(
        random_range=random_params["cars_position_noise"],  # 无人车位置噪声, 1维向量
    )
    # ===========================

    # 初始化DQN强化学习参数
    dqn_params = {
        'obs_dim': 12,  # 观察维度
        'action_dim': 4,  # 动作维度
        'hidden_dim': args.hidden_dim,  # 隐藏层维度
        'lr': args.lr,  # 学习率
        'gamma': args.gamma,  # 折扣因子
        'epsilon': args.epsilon,  # 测试时使用较小的探索率
        'epsilon_min': args.epsilon_min,  # 最小探索率
        'epsilon_decay': args.epsilon_decay,  # 探索率衰减
        'tau': args.tau,  # 目标网络软更新参数
        'buffer_size': args.buffer_size,  # 经验回放缓冲区大小
        'batch_size': args.batch_size,  # 批处理大小
        'warmup_episodes': args.warmup_episodes  # 预热回合数
    }

    # 初始化多智能体DQN强化学习
    trainer = MultiAgentDQN(blue_agents, dqn_params)

    # 加载预训练模型
    try:
        trainer.agent.load_model(args.model_path)
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("使用随机初始化的模型进行测试")

    # 加载预训练的模型（Actor），用于无人车的路径规划。
    actors_cur1 = [None for _ in range(int(iifds.numberofcar / 2))]
    actors_cur2 = [None for _ in range(int(iifds.numberofcar / 2))]
    for i in range(int(iifds.numberofcar / 2)):
        actors_cur1[i] = torch.load('../Path_model/Actor.%d.pkl' % 0, map_location=device)
        actors_cur2[i] = torch.load('../Path_model/Actor.%d.pkl' % 1, map_location=device)

    # 记录测试结果
    episode_rewards = []
    episode_kills = []
    episode_deaths = []
    episode_survival_rates = []
    episode_wins = []
    episode_draws = []
    episode_losses = []

    # 开始测试
    print(f"开始测试，共 {num_episodes} 个回合...")

    for episode in range(num_episodes):
        # 重置环境
        env = Battle(args, X_range, Y_range, obsCenter, obsR)  # Battle 类实例化环境，调用 env.reset() 重置环境状态。

        iifds = IIFDS(X_range, Y_range, obsCenter, obsR)  # 使用 IIFDS 来处理无人车行为的细节（例如任务分配、路径规划等）。
        # 初始化状态变量
        total_reward = 0

        # 初始化无人车的当前位置q，上一位置qBefore，当前速度v，初始位置start，目标goal，障碍物位置obsCenter，障碍物速度Vobs。
        q = []
        qBefore = []
        v = []

        start = iifds.start  # 当前为三维坐标，若考虑二维空间，则将第三维坐标固定。
        goal = iifds.goal

        vObs = []
        obs_num = len(obsCenter)
        for i in range(obs_num):
            vObs.append(np.array([0, 0, 0], dtype=float))  # 设置为静态障碍物。
        # np.savetxt('./Data_csv/obsCenter.csv', obsCenter, delimiter=',')
        for i in range(iifds.numberofcar):
            q.append(start[i])
            qBefore.append([None, None, None])
            v.append(0.001 * q[i])  # 初始速度置为0。
        # 使用 globals() 将每个无人车的路径和目标动态赋值给 pathX 和 goalX。
        path = []
        target = []

        for i in range(iifds.numberofcar):
            path.append(start[i][0:2].reshape(1, -1))
            target.append(start[i][0:2].reshape(1, -1))

        # 将 path 和 goal 转换为单独的变量
        for i in range(iifds.numberofcar):
            globals()[f'path{i + 1}'] = path[i]
            globals()[f'goal{i + 1}'] = target[i]

        task_index = np.ones(iifds.numberofcar) * -3  # 表示当前时刻各无人车的任务目标，-3表示搜索，-2表示逃逸，-1表示支援，0表示追击
        flag_car = np.zeros(iifds.numberofcar)  # 表示当前时刻各无人车的存活情况，0表示存活，1表示死亡
        missle_index = np.ones(iifds.numberofcar) * iifds.missle_num  # 存放每一轮各无人车的子弹剩余数量
        fill_index = np.zeros(iifds.numberofcar)  # 存放所有时刻各无人车的子弹填充情况
        flag_fill = np.zeros(iifds.numberofcar)  # 表示当前时刻子弹是否填充完成，填充完毕为1，否则为0
        HP_index = np.ones(iifds.numberofcar) * iifds.HP_num  # 表示当前时刻各无人车的血量剩余情况
        fire_car = np.ones(iifds.numberofcar) * -1

        ta_index = task_index.reshape(1, -1)  # 存放所有时刻各无人车的任务目标
        dead_index = flag_car.reshape(1, -1)  # 存放所有时刻各无人车的存活情况
        total_missle_index = missle_index.reshape(1, -1)  # 表示所有时刻各无人车的血量剩余情况
        total_HP_index = HP_index.reshape(1, -1)  # 表示所有时刻各无人车的血量剩余情况

        prev_ally_alive = int(iifds.numberofcar / 2)  # 初始己方数量
        prev_enemy_alive = int(iifds.numberofcar / 2)  # 初始敌军数量
        prev_ally_HP = sum(HP_index[:int(iifds.numberofcar / 2)])
        prev_enemy_HP = sum(HP_index[int(iifds.numberofcar / 2):])
        initial_blue_alive = int(iifds.numberofcar / 2)  # 初始蓝方数量

        # 初始化观察
        obs = [None] * int(iifds.numberofcar / 2)
        # 记录每回合的动作分布
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0:搜索, 1:逃跑, 2:追击, 3:支援

        for step in range(max_steps):
            # 路径拼接
            pos_b = []
            pos_r = []
            # 创建 pos_b_all 和 pos_r_all
            pos_b_all = [[j + 1, globals()[f'path{j + 1}']] for j in range(int(iifds.numberofcar / 2))]  # 蓝队路径
            pos_r_all = [[j + 1, globals()[f'path{int(j + 1 + iifds.numberofcar / 2)}']] for j in
                         range(int(iifds.numberofcar / 2))]  # 红队路径

            # 保存所有存放无人车的路径点用于轨迹预测
            for j in range(iifds.numberofcar):
                if flag_car[j] == 0:
                    if j < iifds.numberofcar / 2:
                        pos_b.append(pos_b_all[j])
                    else:
                        pos_r.append(pos_r_all[j - int(iifds.numberofcar / 2)])

            perception_q.updata_actrual_q(q)
            random_q = perception_q.get_observation()
            random_q, q = q, random_q

            all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei = iifds.detect(
                q, v, flag_car, ta_index, HP_index,
                obsCenter)

            # 获取存活的蓝方智能体
            alive_blue = iifds.get_alive_blue_agents(flag_car)

            # 获取每个智能体的观察
            for car_id in range(int(iifds.numberofcar / 2)):
                obs[car_id], _, _ = iifds.get_obs(car_id, alive_blue, all_opp, q, v, flag_car, HP_index)

            # 获取动作并更新任务目标
            actions = trainer.get_actions(obs, eval_mode=True)  # 使用评估模式

            # 记录动作分布
            for action in actions:
                action_counts[action] += 1

            goal, task_index = iifds.assign(q, v, flag_car, goal, missle_index, step,
                                            pos_b, pos_r, ta_index, obsCenter, all_opp, all_nei, all_nei_c2e,
                                            all_close_opp, all_close_nei, actions)
            print(
                f"*********************Episode {episode}/{num_episodes}| Step {step}/{max_steps}| Reward: {total_reward:.2f}**********************")

            # 更新任务索引
            ta_index = np.vstack((ta_index, task_index))

            # 路径规划
            obsCenterNext = obsCenter
            vObsNext = vObs
            # 根据当前位置、目标位置、障碍物位置，规划避障路径，输出为下一时刻的速度矢量（包括大小和方向）
            vNext = iifds.getvNext(q, v, obsCenter, vObs, qBefore, goal, flag_car, arglist, actors_cur1, actors_cur2)
            random_q, q = q, random_q

            qNext, _ = physical_law.get_qvNext(q, v, vNext, flag_car)
            # 计算伤亡情况
            for j in range(iifds.numberofcar):
                if flag_car[j] == 1:
                    vNext[j] = np.array([0, 0, 0])
                    missle_index[j] = 0
                    fire_car[j] = -1
                else:
                    if missle_index[j] != 0:
                        for k in range(int(iifds.numberofcar / 2)):
                            if j < iifds.numberofcar / 2:
                                k += int(iifds.numberofcar / 2)
                            if flag_car[k] == 0 and iifds.distanceCost(q[k],
                                                                       q[j]) < iifds.threshold and iifds.cos_cal(
                                q[k] - q[j],
                                v[j]) > np.cos(
                                iifds.hit_angle / 2) and iifds.detect_obs(q[j], q[k], obsCenter) == 0:  # 目标小于开火范围
                                missle_index[j] -= 1
                                fire_car[j] = k
                                break
                    else:  # 若子弹未填充完毕，则继续填充，且判断打击是否有效
                        fill_index[j] += iifds.vel_fill_missle
                        if fill_index[j] >= iifds.missle_num:  # 若子弹填充完毕，则可以继续参与打击
                            flag_fill[j] = 0
                            fill_index[j] = 0
                            missle_index[j] = iifds.missle_num
                        if fire_car[j] >= 0:
                            if iifds.cos_cal(qBefore[int(fire_car[j])] - qBefore[j],
                                             q[int(fire_car[j])] - qBefore[j]) > np.cos(
                                    np.arctan(iifds.carR / iifds.distanceCost(qBefore[int(fire_car[j])], qBefore[j]))):
                                if HP_index[int(fire_car[j])] > 0:
                                    HP_index[int(fire_car[j])] -= 1
                                    if HP_index[int(fire_car[j])] == 0:
                                        flag_car[int(fire_car[j])] = 1
                            fire_car[j] = -1
            qNext, vNext = physical_law.get_qvNext(q, v, vNext, flag_car)
            # 为了死亡的时候仍然有朝向显示，增加微小的速度值：
            for j in range(iifds.numberofcar):
                if flag_car[j] == 1:
                    vNext[j] = 0.0001 * np.array(v[j]) / np.linalg.norm(np.array(v[j]))
                    qNext[j] = np.array(q[j]) + vNext[j]
            env.render(q, v, fire_car, flag_car, HP_index, iifds.HP_num, missle_index,
                       iifds.missle_num)  # 画出当前一时刻的无人车的位置速度、血量、弹药

            # 计算奖励
            current_ally_alive = sum(1 - flag_car[:int(iifds.numberofcar / 2)])
            current_enemy_alive = sum(1 - flag_car[int(iifds.numberofcar / 2):])
            current_ally_HP = sum(HP_index[:int(iifds.numberofcar / 2)])
            current_enemy_HP = sum(HP_index[int(iifds.numberofcar / 2):])
            kill_ally = prev_ally_alive - current_ally_alive
            kill_enemy = prev_enemy_alive - current_enemy_alive
            HP_loss_ally = prev_ally_HP - current_ally_HP
            HP_loss_enemy = prev_enemy_HP - current_enemy_HP
            global_reward = (kill_enemy - kill_ally) * 10 * iifds.HP_num + (HP_loss_enemy - HP_loss_ally) * 10
            prev_ally_alive = current_ally_alive
            prev_enemy_alive = current_enemy_alive
            prev_ally_HP = current_ally_HP
            prev_enemy_HP = current_enemy_HP
            rewards = getReward1(iifds, actions, obs, global_reward, all_opp, all_nei_c2e, dqn_params['obs_dim'])

            qBefore = q
            q = qNext
            v = vNext
            obsCenter = obsCenterNext
            vObs = vObsNext

            # 信息保存
            for j in range(iifds.numberofcar):
                path_var = globals().get(f'path{j + 1}')
                goal_var = globals().get(f'goal{j + 1}')

                path_var = np.vstack((path_var, q[j][0:2]))
                goal_var = np.vstack((goal_var, goal[j][0:2]))

                # 更新全局变量
                globals()[f'path{j + 1}'] = path_var
                globals()[f'goal{j + 1}'] = goal_var
            ta_index = np.vstack((ta_index, task_index))
            dead_index = np.vstack((dead_index, flag_car))
            total_missle_index = np.vstack((total_missle_index, missle_index))
            total_HP_index = np.vstack((total_HP_index, HP_index))

            # 获取下一个观察
            all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei = iifds.detect(q, v, flag_car, ta_index,
                                                                                       HP_index, obsCenter)
            next_obs = [None] * blue_agents
            alive_blue = iifds.get_alive_blue_agents(flag_car)

            for car_id in range(blue_agents):
                next_obs[car_id], _, _ = iifds.get_obs(car_id, alive_blue, all_opp, q, v, flag_car, HP_index)

            # 检查是否结束
            blue_dones, _, global_done = iifds.get_agent_dones(flag_car)

            # 更新状态和奖励
            obs = next_obs
            total_reward += sum(rewards)

            if global_done:
                break

        # 计算回合结果
        current_blue_alive = sum(1 - flag_car[:int(iifds.numberofcar / 2)])
        current_red_alive = sum(1 - flag_car[int(iifds.numberofcar / 2):])
        blue_survival_rate = current_blue_alive / initial_blue_alive
        enemy_kills = initial_blue_alive - current_red_alive
        blue_deaths = initial_blue_alive - current_blue_alive

        # 判断胜负
        blue_win = current_blue_alive > current_red_alive
        red_win = current_blue_alive < current_red_alive
        draw = current_blue_alive == current_red_alive
        episode_wins.append(1 if blue_win else 0)
        episode_draws.append(1 if draw else 0)
        episode_losses.append(1 if red_win else 0)

        # 记录结果
        episode_rewards.append(total_reward)
        episode_kills.append(enemy_kills)
        episode_deaths.append(blue_deaths)
        episode_survival_rates.append(blue_survival_rate)

        # 输出回合结果
        print(
            f"===========Episode {episode}, Total Reward: {total_reward:.2f}, Kills: {enemy_kills}, Deaths: {blue_deaths}, Survival Rate: {blue_survival_rate:.2f}, {'蓝方胜利' if blue_win else '红方胜利'}===========")
        print(
            f"动作分布: 搜索={action_counts[0]}, 逃跑={action_counts[1]}, 追击={action_counts[2]}, 支援={action_counts[3]}")

        if env is not None:
            env.close()  # Ensure the viewer is closed after each episode

    # 计算平均结果
    avg_reward = np.mean(episode_rewards)
    avg_kills = np.mean(episode_kills)
    avg_deaths = np.mean(episode_deaths)
    avg_survival_rate = np.mean(episode_survival_rates)
    blue_win_rate = np.mean(episode_wins)
    red_win_rate = np.mean(episode_losses)
    draw_rate = np.mean(episode_draws)

    # 计算击杀死亡比
    kill_death_ratio = avg_kills / (avg_deaths + 1e-6)  # 避免除以零

    print(f"\n测试结果总结:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均击杀数: {avg_kills:.2f}")
    print(f"平均死亡数: {avg_deaths:.2f}")
    print(f"击杀死亡比: {kill_death_ratio:.2f}")
    print(f"平均存活率: {avg_survival_rate:.2f}")
    print(f"蓝方胜率: {blue_win_rate:.2f}")
    print(f"红方胜率: {red_win_rate:.2f}")
    print(f"平局率: {draw_rate:.2f}")

    results = {
        "avg_reward": avg_reward,
        "avg_kills": avg_kills,
        "avg_deaths": avg_deaths,
        "kill_death_ratio": kill_death_ratio,
        "avg_survival_rate": avg_survival_rate,
        "blue_win_rate": blue_win_rate,
        "episode_rewards": episode_rewards,
        "episode_kills": episode_kills,
        "episode_deaths": episode_deaths,
        "episode_survival_rates": episode_survival_rates,
        "episode_wins": episode_wins,
        "episode_draws": episode_draws,
        "episode_losses": episode_losses
    }

    if not os.path.exists("test_results"):
        os.makedirs("test_results")

    # 保存结果到文本文件
    file_path = os.path.join("test_results", "best_dqn_model_ps.txt")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("dqn_test_results:\n")
        f.write(f"avg_reward: {avg_reward:.2f}\n")
        f.write(f"avg_kills: {avg_kills:.2f}\n")
        f.write(f"avg_deaths: {avg_deaths:.2f}\n")
        f.write(f"kill_death_ratio: {kill_death_ratio:.2f}\n")
        f.write(f"avg_survival_rate: {avg_survival_rate:.2f}\n")
        f.write(f"blue_win_rate: {blue_win_rate:.2f}\n")
        f.write(f"red_win_rate: {red_win_rate:.2f}\n")
        f.write(f"draw_rate: {draw_rate:.2f}\n\n")

        # 添加DQN强化学习参数
        f.write("DQN强化学习参数:\n")
        f.write(f"obs_dim: {dqn_params['obs_dim']}\n")
        f.write(f"action_dim: {dqn_params['action_dim']}\n")
        f.write(f"hidden_dim: {dqn_params['hidden_dim']}\n")
        f.write(f"lr: {dqn_params['lr']}\n")
        f.write(f"gamma: {dqn_params['gamma']}\n")
        f.write(f"epsilon: {dqn_params['epsilon']}\n")
        f.write(f"epsilon_min: {dqn_params['epsilon_min']}\n")
        f.write(f"epsilon_decay: {dqn_params['epsilon_decay']}\n")
        f.write(f"tau: {dqn_params['tau']}\n")
        f.write(f"buffer_size: {dqn_params['buffer_size']}\n")
        f.write(f"batch_size: {dqn_params['batch_size']}\n")
        f.write(f"warmup_episodes: {dqn_params['warmup_episodes']}\n\n")

        # 添加测试参数
        f.write("测试参数:\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"max_steps: {max_steps}\n")
        f.write(f"model_path: {args.model_path}\n")
        f.write(f"num_RCARs: {args.num_RCARs}\n")
        f.write(f"num_BCARs: {args.num_BCARs}\n")
        f.write(f"num_Bcars: {args.num_Bcars}\n")
        f.write(f"detect_range: {args.detect_range}\n")
        f.write(f"attack_range_B: {args.attack_range_B}\n")
        f.write(f"attack_range_R: {args.attack_range_R}\n")
        f.write(f"attack_angle_BR: {args.attack_angle_BR}\n")
        f.write(f"sensor_range_B_l: {args.sensor_range_B_l}\n")
        f.write(f"sensor_range_B_w: {args.sensor_range_B_w}\n")
        f.write(f"sensor_angle_B: {args.sensor_angle_B}\n")

    print(f"测试结果已保存到 {file_path}")
    return results


if __name__ == "__main__":
    obsCenter = pd.read_csv('../Data_csv/obsCenter.csv').to_numpy()
    obsR = 0.3
    X_range, Y_range = 15, 8
    args = get_args()
    results = test_model(args, obsCenter)
