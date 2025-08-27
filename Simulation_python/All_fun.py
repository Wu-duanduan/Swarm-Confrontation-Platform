#!/usr/bin/python
import os
import shutil

import numpy as np
import random
import heapq

import pandas as pd

from find_enemy_area import FindEnemyArea
import math
import torch
import cv2
import re
import matplotlib.pyplot as plt


class IIFDS:
    """使用IIFDS类训练时每次必须reset"""

    def __init__(self, X_range=None, Y_range=None, obsCenter=None, obsR=None,
                 vlm_answer_dir=None,
                 vlm_prompt_dir=None,
                 llm_answer_dir=None,
                 llm_prompt_dir=None):
        """基本参数："""
        self.vlm_answer_dir = vlm_answer_dir
        self.vlm_prompt_dir = vlm_prompt_dir
        self.llm_answer_dir = llm_answer_dir
        self.llm_prompt_dir = llm_prompt_dir
        self.V1 = 0.8  # 速度大小的最大值限制
        self.V2 = 0.8
        self.threshold = 1.5  # 最大打击距离阈值，在该打击距离下，无人车无法隔墙打击
        self.threshold2 = 0.4  # 搜索任务的到达距离阈值
        self.threshold3 = 0.4  # 逃跑任务的到达距离阈值
        self.stepSize = 0.1  # 时间间隔步长
        self.lam = 0.4  # 避障参数，越大考虑障碍物速度越明显
        self.numberofcar = 24  # 无人车数量
        self.carR = 0.3  # 无人车半径
        self.num_com = 1  # 路径规划时考虑的邻居数量
        self.obsR = obsR  # 障碍物半径
        self.R_1 = 5  # 针对敌军的感知半径
        self.R_2 = 5  # 针对友军的通信半径
        self.missle_num = 1  # 最大子弹填充数量
        self.hit_angle = np.pi / 2  # 子弹攻击角度
        self.hit_rate = 0.5  # 子弹命中概率
        self.vel_missle = 2
        self.observe_angle = np.pi  # 观测角度
        self.HP_num = 3  # 初始生命值
        self.end_predict = 0  # 开始预测的回合
        self.vel_fill_missle = self.missle_num / 10  # 子弹填充速度
        self.x_max = X_range  # 场地边界
        self.y_max = Y_range
        # 初始位置设置
        self.start = []
        self.obsCenter = obsCenter
        while len(self.start) < self.numberofcar:
            car_xyz = np.array([np.random.randint(-(self.x_max - 1), (self.x_max - 1)),
                                np.random.randint(-(self.y_max - 1), (self.y_max - 1)),
                                0.8 * 2 / 3], dtype=float)
            flag = 0
            for j in range(len(self.start)):
                if self.distanceCost(self.start[j], car_xyz) < 2 * self.carR + 1:
                    flag = 1
            for j in range(len(self.obsCenter)):
                if self.distanceCost(self.obsCenter[j], car_xyz) < self.carR + self.obsR + 1:
                    flag = 1
            if flag == 0:
                self.start.append(car_xyz)

        self.ass = list(range(0, self.numberofcar))

        self.safePos = []
        self.goal = []
        for i in range(len(self.ass)):
            self.goal.append(self.start[self.ass[i]])

        self.timelog = 0  # 时间，用来计算动态障碍的位置
        self.timeStep = 0.1

        self.xmax = 10 / 180 * np.pi  # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10 / 180 * np.pi  # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100 / 180 * np.pi  # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角

        self.vObs = None
        self.vObsNext = None

    def get_agent_dones(self, flag_car):
        """
        计算双方智能体的终止状态
        Returns:
            tuple: (blue_dones, red_dones, global_done)
                blue_dones: 蓝方各智能体终止状态列表（死亡为True）
                red_dones: 红方各智能体终止状态列表
                global_done: 是否全局终止（一方全灭）
        """
        assert len(flag_car) == self.numberofcar, "输入数组长度必须为10"

        # 提取双方状态
        blue_flags = flag_car[:int(self.numberofcar / 2)]
        red_flags = flag_car[int(self.numberofcar / 2):]

        # 计算个体终止状态
        blue_dones = (blue_flags == 1).tolist()
        red_dones = (red_flags == 1).tolist()

        # 全局终止条件：蓝方全灭或红方全灭
        global_done = all(blue_dones) or all(red_dones)

        return blue_dones, red_dones, global_done

    def relative_position_and_orientation(self, q, v, i, j):
        """
        计算无人机j相对于无人机i的相对坐标和相对朝向角

        返回:
        relative_pos: 相对坐标（2D）
        relative_angle: 相对朝向角（弧度）
        """

        # 计算相对坐标（只取x和y）
        relative_pos = q[j][:2] - q[i][:2]

        # 计算无人机i的朝向角（假设朝向与速度方向一致）
        angle_i = np.arctan2(v[i][1], v[i][0])

        # 计算无人机j的朝向角
        angle_j = np.arctan2(v[j][1], v[j][0])

        # 计算相对朝向角
        relative_angle = angle_j - angle_i

        # 确保相对朝向角在 -pi 到 pi 之间
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        return relative_pos, relative_angle

    def calculate_situation(self, q, v, hp, i, j):
        """
        计算红蓝双方的态势（i对j）
        返回值：distance, situation，其中situation范围[-1,1]，正值表示car_id优势
        """
        # 获取双方的坐标（忽略z轴）
        q_i = [q[i][0], q[i][1]]
        q_j = [q[j][0], q[j][1]]

        # 计算相对位置向量（从i指向j）
        dx = q_j[0] - q_i[0]
        dy = q_j[1] - q_i[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 计算i的方向对准得分
        v_i = [v[i][0], v[i][1]]
        vi_len = math.sqrt(v_i[0] ** 2 + v_i[1] ** 2)
        if vi_len == 0:
            cos_i = 0.0
        else:
            dot_i = dx * v_i[0] + dy * v_i[1]
            cos_i = dot_i / (vi_len * distance)
            cos_i = max(min(cos_i, 1.0), -1.0)  # 限制在[-1,1]

        # 计算j的方向对准得分（从j指向i的向量为 -dx, -dy）
        v_j = [v[j][0], v[j][1]]
        vj_len = math.sqrt(v_j[0] ** 2 + v_j[1] ** 2)
        if vj_len == 0:
            cos_j = 0.0
        else:
            dot_j = (-dx) * v_j[0] + (-dy) * v_j[1]
            cos_j = dot_j / (vj_len * distance)
            cos_j = max(min(cos_j, 1.0), -1.0)

        # 态势计算（蓝方得分 - 红方得分，归一化到[-1,1]）
        situation = (cos_i - cos_j) / 2.0

        hp_count = hp[j] - hp[i]
        return distance, situation, hp_count

    def get_alive_blue_agents(self, flag_car):
        """获取蓝方存活的无人机编号"""
        return [i for i in range(int(self.numberofcar / 2)) if flag_car[i] == 0]

    def get_nearest_cars(self, car_id, alive_blue, opp, positions):
        """
        获取距离指定车辆最近的两辆存活蓝方车和两辆存活红方车

        返回:
        nearest_blue: 最近的两辆存活蓝方车的编号列表
        nearest_red: 最近的两辆存活红方车的编号列表
        """
        # 计算与其他蓝方车的距离
        blue_distances = []
        for other_id in alive_blue:
            if other_id != car_id:
                dist = np.linalg.norm(positions[car_id] - positions[other_id])
                blue_distances.append((other_id, dist))

        # 计算与红方车的距离
        red_distances = []
        for other_id in opp:
            dist = np.linalg.norm(positions[car_id] - positions[other_id])
            red_distances.append((other_id, dist))

        # 按距离排序并选择最近的两辆车
        blue_distances.sort(key=lambda x: x[1])
        red_distances.sort(key=lambda x: x[1])

        nearest_blue = [id for id, _ in blue_distances[:2]]
        nearest_red = [id for id, _ in red_distances[:2]]
        # 如果蓝方或红方存活车辆不足两辆，用-1填充
        nearest_blue = (nearest_blue + [-1, -1])[:2]
        nearest_red = (nearest_red + [-1, -1])[:2]

        return nearest_blue, nearest_red

    def get_obs(self, car_id, alive_blue, all_opp, q, v, flag_car, hp):
        obs = []
        obs_mask = []  # 添加掩码
        uncertainty = []  # 添加不确定性信息

        if flag_car[car_id] == 0:
            nearest_blue, nearest_red = self.get_nearest_cars(car_id, alive_blue, all_opp[car_id], q)

            # 处理友军信息
            for j in nearest_blue:
                if j != -1:
                    distance, situation, hp_count = self.calculate_situation(q, v, hp, car_id, j)  # 计算car_id对j的距离和态势
                    # 友军态势计算
                    obs.extend([float(distance), float(situation), float(hp_count)])
                    obs_mask.extend([1.0, 1.0, 1.0])  # 观测到的位置标记为1
                    # 根据距离计算不确定性
                    uncertainty.extend([1.0 / (1.0 + distance), 1.0 / (1.0 + distance), 1.0 / (1.0 + distance)])
                else:
                    obs.extend([0.0, 0.0, 0.0])
                    obs_mask.extend([0.0, 0.0, 0.0])
                    uncertainty.extend([0.0, 0.0, 0.0])  # 未观测到的不确定性为0

            # 处理敌军信息
            for k in nearest_red:
                if k != -1:
                    distance, situation, hp_count = self.calculate_situation(q, v, hp, car_id, k)  # 计算car_id对k的距离和态势
                    obs.extend([float(distance), float(situation), float(hp_count)])
                    obs_mask.extend([1.0, 1.0, 1.0])
                    # 根据距离计算不确定性
                    uncertainty.extend([1.0 / (1.0 + distance), 1.0 / (1.0 + distance), 1.0 / (1.0 + distance)])
                else:
                    obs.extend([0.0, 0.0, 0.0])
                    obs_mask.extend([0.0, 0.0, 0.0])
                    uncertainty.extend([0.0, 0.0, 0.0])  # 未观测到的不确定性为0
        else:
            obs = [0.0] * 12
            obs_mask = [0.0] * 12
            uncertainty = [0.0] * 12
        return obs, obs_mask, uncertainty

    def detect_obs(self, carPos1, carPos2, obsCenter):
        for k in range(len(obsCenter)):
            if len(self.line_intersect_circle((obsCenter[k][0], obsCenter[k][1], self.obsR),
                                              (carPos1[0], carPos1[1]),
                                              (carPos2[0], carPos2[1]))) == 2:
                return 1
        return 0
    
    def detect(self, carPos, carVel, flag_car, ta_index, HP_index, obsCenter):
        all_opp = []
        all_nei_c2e = []
        all_nei = []
        all_close_opp = []
        all_close_nei = []
        for i in range(self.numberofcar):
            distance1 = np.ones([1, int(self.numberofcar)]) * np.inf
            distance2 = np.ones([1, int(self.numberofcar)]) * np.inf
            opp = []
            nei = []
            nei_c2e = []
            if i < int(self.numberofcar / 2) and flag_car[i] == 0:
                for j in range(self.numberofcar):
                    if j != i and j >= int(self.numberofcar / 2):  # 敌方判断
                        if flag_car[j] == 0:  # 存活判断
                            if self.distanceCost(carPos[i], carPos[j]) < self.R_1:  # 感知判断
                                flag_detected = self.detect_obs(carPos[i], carPos[j], obsCenter)
                                if flag_detected == 0:
                                    opp.append(j)  # 存放能感知到的敌军
                                    distance1[0][j] = self.distanceCost(carPos[i], carPos[j])  # 敌军信息包括位置、血量
                    elif j != i and j < int(self.numberofcar / 2):  # 友方判断
                        if flag_car[j] == 0:  # 存活判断
                            if self.distanceCost(carPos[i], carPos[j]) < self.R_2:  # 感知判断
                                nei.append(j)  # 存放能感知到的友军
                                if ta_index[-1][j] == 0 or ta_index[-1][j] == -2:  # 记录追击和逃跑的友军
                                    nei_c2e.append(j)
                                    distance2[0][j] = self.distanceCost(carPos[i], carPos[j])
            elif i >= int(self.numberofcar / 2) and flag_car[i] == 0:
                for j in range(self.numberofcar):
                    if j != i and j < int(self.numberofcar / 2):
                        if flag_car[j] == 0:
                            if self.distanceCost(carPos[i], carPos[j]) < self.R_1:  # 感知判断
                                flag_detected = self.detect_obs(carPos[i], carPos[j], obsCenter)
                                if flag_detected == 0:
                                    opp.append(j)
                                    distance1[0][j] = self.distanceCost(carPos[i], carPos[j])
                    elif j != i and j >= int(self.numberofcar / 2):
                        if flag_car[j] == 0:
                            if self.distanceCost(carPos[i], carPos[j]) < self.R_2:  # 感知判断
                                nei.append(j)
                                if ta_index[-1][j] == 0 or ta_index[-1][j] == -2:  # 记录追击、逃跑的友军
                                    nei_c2e.append(j)
                                    distance2[0][j] = self.distanceCost(carPos[i], carPos[j])
            car_catch = heapq.nsmallest(1, distance1[0])
            index1 = list(map(distance1[0].tolist().index, car_catch))
            car_contact = heapq.nsmallest(1, distance2[0])
            index2 = list(map(distance2[0].tolist().index, car_contact))
            all_opp.append(opp)
            all_nei.append(nei)
            all_nei_c2e.append(nei_c2e)
            all_close_opp.append(index1[0])
            all_close_nei.append(index2[0])
        return all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei

    def assign(self, carPos, carVel, flag_car, goal, missle_index, step, pos_b, pos_r, ta_index,
               obsCenter, all_opp, all_nei, all_nei_c2e, all_close_opp, all_close_nei, actions):
        task_index = []

        carPos_copy = np.copy(carPos)
        carVel_copy = np.copy(carVel)
        goal_copy = np.copy(goal)

        for i in range(self.numberofcar):
            if flag_car[i] == 1:  # 如果死亡
                task_index.append(-3)  # 搜索
            else:
                if i < int(self.numberofcar / 2):
                    if actions[i] == 2:
                        if len(all_opp[i]) != 0:
                            goal_copy[i] = carPos_copy[all_close_opp[i]]
                        task_index.append(0)  # 追击
                        print(f"{i}号车：追击")
                    elif actions[i] == 1:
                        if len(all_opp[i]) != 0:
                            goal_copy[i] = carPos_copy[i]
                            for j in range(len(all_opp[i])):
                                goal_copy[i] += (carPos_copy[i] - carPos_copy[all_opp[i][j]]) / self.distanceCost(
                                    carPos_copy[i],
                                    carPos_copy[
                                        all_opp[i][j]]) ** 2 * (
                                                        self.distanceCost(carPos_copy[i],
                                                                          carPos_copy[all_opp[i][j]]) + 1)
                            goal_copy[i][0] += ((carPos_copy[i][0] - self.x_max) + 10) / (
                                    carPos_copy[i][0] - self.x_max)
                            goal_copy[i][0] += ((carPos_copy[i][0] - -self.x_max) + 10) / (
                                    carPos_copy[i][0] - -self.x_max)
                            goal_copy[i][1] += ((carPos_copy[i][1] - self.y_max) + 10) / (
                                    carPos_copy[i][1] - self.y_max)
                            goal_copy[i][1] += ((carPos_copy[i][1] - -self.y_max) + 10) / (
                                    carPos_copy[i][1] - -self.y_max)
                        task_index.append(-2)  # 逃逸
                        print(f"{i}号车：逃跑")
                    elif actions[i] == 0:
                        if step > self.end_predict and (
                                ta_index[-1][i] != -3 or step % 5 == 0 or self.distanceCost(goal_copy[i],
                                                                                            carPos_copy[
                                                                                                i]) < self.threshold2):
                            finder = FindEnemyArea(pos_r, obsCenter, self.timeStep, self.obsR + self.carR, self.x_max,
                                                   self.y_max)
                            temp = finder.predict_trajectory(10)
                            try:
                                goal_copy[i][0:2] = finder.find_nearest_center(temp, carPos_copy[i][0:2])
                            except Exception as e:
                                pass
                        task_index.append(-3)  # 搜索
                        print(f"{i}号车：搜索")
                    elif actions[i] == 3:
                        if len(all_nei_c2e[i]) != 0:  # 存在逃跑或追击的友军
                            goal_copy[i] = carPos_copy[all_close_nei[i]]
                        task_index.append(-1)  # 支援
                        print(f"{i}号车：支援")
                else:
                    if missle_index[i] == 0:
                        goal_copy[i] = carPos_copy[i]
                        for j in range(len(all_opp[i])):
                            goal_copy[i] += (carPos_copy[i] - carPos_copy[all_opp[i][j]]) / self.distanceCost(
                                carPos_copy[i],
                                carPos_copy[
                                    all_opp[i][j]]) ** 2 * (
                                                    self.distanceCost(carPos_copy[i], carPos_copy[all_opp[i][j]]) + 1)
                        goal_copy[i][0] += ((carPos_copy[i][0] - self.x_max) + 10) / (carPos_copy[i][0] - self.x_max)
                        goal_copy[i][0] += ((carPos_copy[i][0] - -self.x_max) + 10) / (carPos_copy[i][0] - -self.x_max)
                        goal_copy[i][1] += ((carPos_copy[i][1] - self.y_max) + 10) / (carPos_copy[i][1] - self.y_max)
                        goal_copy[i][1] += ((carPos_copy[i][1] - -self.y_max) + 10) / (carPos_copy[i][1] - -self.y_max)
                        task_index.append(-2)  # 逃逸
                    else:
                        if len(all_opp[i]) != 0:
                            if self.cos_cal(carVel_copy[i], carPos_copy[all_close_opp[i]] - carPos_copy[i]) >= self.cos_cal(
                                    carVel_copy[all_close_opp[i]], -carPos_copy[all_close_opp[i]] + carPos_copy[i]):
                                goal_copy[i] = carPos_copy[all_close_opp[i]]
                                task_index.append(0)  # 追击
                            else:
                                goal_copy[i] = carPos_copy[i]
                                for j in range(len(all_opp[i])):
                                    goal_copy[i] += (carPos_copy[i] - carPos_copy[all_opp[i][j]]) / self.distanceCost(
                                        carPos_copy[i],
                                        carPos_copy[
                                            all_opp[i][j]]) ** 2 * (
                                                            self.distanceCost(carPos_copy[i],
                                                                              carPos_copy[all_opp[i][j]]) + 1)
                                goal_copy[i][0] += ((carPos_copy[i][0] - self.x_max) + 10) / (
                                            carPos_copy[i][0] - self.x_max)
                                goal_copy[i][0] += ((carPos_copy[i][0] - -self.x_max) + 10) / (
                                            carPos_copy[i][0] - -self.x_max)
                                goal_copy[i][1] += ((carPos_copy[i][1] - self.y_max) + 10) / (
                                            carPos_copy[i][1] - self.y_max)
                                goal_copy[i][1] += ((carPos_copy[i][1] - -self.y_max) + 10) / (
                                            carPos_copy[i][1] - -self.y_max)
                                task_index.append(-2)  # 逃逸
                        else:
                            if len(all_nei_c2e[i]) != 0:  # 存在逃跑或追击的友军
                                goal_copy[i] = carPos_copy[all_close_nei[i]]
                                task_index.append(-1)  # 支援
                            else:
                                if step > self.end_predict and (
                                        ta_index[-1][i] != -3 or step % 5 == 0 or self.distanceCost(goal_copy[i],
                                                                                                    carPos_copy[
                                                                                                        i]) < self.threshold2):
                                    finder = FindEnemyArea(pos_b, obsCenter, self.timeStep, self.obsR + self.carR,
                                                           self.x_max, self.y_max)
                                    temp = finder.predict_trajectory(10)
                                    try:
                                        goal_copy[i][0:2] = finder.find_nearest_center(temp, carPos_copy[i][0:2])
                                    except Exception as e:
                                        pass
                                task_index.append(-3)  # 搜索
        return goal_copy, task_index

    def cos_cal(self, a, b):
        return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def calDynamicState(self, carPos, carVel, obsPos, obsVel, obs_num, goal, flag_car):
        """强化学习模型获得的state。"""
        dic = {}
        for i in range(len(carPos)):
            dic.update({f'car{i + 1}': []})
        s = []
        for i in range(self.numberofcar):
            s1 = goal[i] - carPos[i]
            s.append(s1)
        # 不仅考虑到观测障碍物 额外还能观测邻居无人车或障碍物
        distance = np.ones([self.numberofcar, int(self.numberofcar + obs_num)]) * np.inf
        for i in range(self.numberofcar):
            for j in range(self.numberofcar + obs_num):
                if j != i and j < int(self.numberofcar):
                    if flag_car[j] == 0:
                        distance[i][j] = self.distanceCost(carPos[i], carPos[j])
                elif j != i and j >= self.numberofcar:
                    distance[i][j] = self.distanceCost(carPos[i][0:2], obsPos[int(j - self.numberofcar)][0:2])

        z = []
        self.car_com = np.zeros([self.numberofcar, self.num_com])
        self.index_com = np.zeros([self.numberofcar, self.num_com])
        for i in range(self.numberofcar):
            self.car_com[i] = heapq.nsmallest(self.num_com, distance[i])
            self.index_com[i] = list(map(distance[i].tolist().index, self.car_com[i]))
            for j in range(int(1)):
                if int(self.index_com[i][j]) < self.numberofcar:
                    z1 = (carPos[int(self.index_com[i][j])] - carPos[i]) * (
                            self.distanceCost(carPos[int(self.index_com[i][j])],
                                              carPos[i]) - 2 * self.carR) / self.distanceCost(
                        carPos[int(self.index_com[i][j])],
                        carPos[i])
                    z.append(z1)
                    z2 = carVel[int(self.index_com[i][j])]
                    z.append(z2)
                else:
                    z1 = (obsPos[int(self.index_com[i][j] - self.numberofcar)] - carPos[i]) * (
                            self.distanceCost(obsPos[int(self.index_com[i][j] - self.numberofcar)][0:2],
                                              carPos[i][0:2]) - (self.carR + self.obsR)) / self.distanceCost(
                        obsPos[int(self.index_com[i][j] - self.numberofcar)][0:2],
                        carPos[i][0:2])
                    z1[2] = 0
                    z.append(z1)
                    z2 = obsVel[int(self.index_com[i][j] - self.numberofcar)]
                    z2[2] = 0
                    z.append(z2)
        for i in range(len(carPos)):
            dic[f'car{i + 1}'].append(np.hstack((s[i], z[2 * i], z[2 * i + 1])))
        return dic

    def calRepulsiveMatrix(self, carPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere(obsCenter, carPos, cylinderR)
        tempD = self.distanceCost(carPos, obsCenter) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(carPos, goal) * tempD))
        T = self.calculateT(obsCenter, carPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix(self, carPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere(obsCenter, carPos, cylinderR)
        T = self.calculateT(obsCenter, carPos, cylinderR)
        partialX = (carPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (carPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = (carPos[2] - obsCenter[2]) * 2 / cylinderR ** 2
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(carPos, obsCenter) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(carPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def calRepulsiveMatrix2(self, carPos, obsCenter, cylinderR, row0, goal):
        n = self.partialDerivativeSphere2(obsCenter, carPos, cylinderR)
        tempD = self.distanceCost(carPos[0:2], obsCenter[0:2]) - cylinderR
        row = row0 * np.exp(1 - 1 / (self.distanceCost(carPos, goal) * tempD))
        T = self.calculateT2(obsCenter, carPos, cylinderR)
        repulsiveMatrix = np.dot(-n, n.T) / T ** (1 / row) / np.dot(n.T, n)[0][0]
        return repulsiveMatrix

    def calTangentialMatrix2(self, carPos, obsCenter, cylinderR, theta, sigma0, goal):
        n = self.partialDerivativeSphere2(obsCenter, carPos, cylinderR)
        T = self.calculateT2(obsCenter, carPos, cylinderR)
        partialX = (carPos[0] - obsCenter[0]) * 2 / cylinderR ** 2
        partialY = (carPos[1] - obsCenter[1]) * 2 / cylinderR ** 2
        partialZ = 0
        tk1 = np.array([partialY, -partialX, 0], dtype=float).reshape(-1, 1)
        tk2 = np.array([partialX * partialZ, partialY * partialZ, -partialX ** 2 - partialY ** 2], dtype=float).reshape(
            -1, 1)
        originalPoint = np.array([np.cos(theta), np.sin(theta), 0]).reshape(1, -1)
        tk = self.trans(originalPoint, tk1.squeeze(), tk2.squeeze(), n.squeeze())
        tempD = self.distanceCost(carPos[0:2], obsCenter[0:2]) - cylinderR
        sigma = sigma0 * np.exp(1 - 1 / (self.distanceCost(carPos, goal) * tempD))
        tangentialMatrix = tk.dot(n.T) / T ** (1 / sigma) / self.calVecLen(tk.squeeze()) / self.calVecLen(n.squeeze())
        return tangentialMatrix

    def getvNext(self, q, v, obsq, obsv, qBefore_all, goal_all, flag_car, arglist, actors_cur1, actors_cur2):

        obsDicq = self.calDynamicState(q, v, obsq, obsv, len(obsq), goal_all, flag_car)  # 相对位置字典

        obs_n1 = obsDicq[f'car{1}']
        for i in range(int(len(q) / 2 - 1)):
            obs_n1 += obsDicq[f'car{i + 2}']

        obs_n2 = obsDicq[f'car{int(self.numberofcar / 2 + 1)}']
        for i in range(int(len(q) / 2 - 1)):
            i += int(self.numberofcar / 2)
            obs_n2 += obsDicq[f'car{i + 2}']

        action_n1 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur1, obs_n1)]
        action_n1 = np.clip(action_n1, arglist.action_limit_min, arglist.action_limit_max)
        action_n1 = action_n1.reshape(-1)

        action_n2 = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                     for agent, obs in zip(actors_cur2, obs_n2)]
        action_n2 = np.clip(action_n2, arglist.action_limit_min, arglist.action_limit_max)
        action_n2 = action_n2.reshape(-1)
        vNext = []
        for i in range(len(q)):
            carPos = q[i]
            goal = goal_all[i]
            qBefore = qBefore_all[i]
            repulsiveMatrix = 0
            tangentialMatrix = 0
            ubar = 0

            if i < self.numberofcar / 2:
                row0 = action_n1[3 * i]
                sigma0 = action_n1[3 * i + 1]
                theta = action_n1[3 * i + 2]
                u = self.initField(carPos, self.V1, goal)
            else:
                row0 = action_n2[3 * (i - int(self.numberofcar / 2))]
                sigma0 = action_n2[3 * (i - int(self.numberofcar / 2)) + 1]
                theta = action_n2[3 * (i - int(self.numberofcar / 2)) + 2]
                u = self.initField(carPos, self.V2, goal)

            for j in range(int(self.num_com)):

                if int(self.index_com[i][j]) < self.numberofcar:
                    repulsiveMatrix += self.calRepulsiveMatrix2(carPos, q[int(self.index_com[i][j])], 2 * self.carR,
                                                                row0,
                                                                goal)
                    tangentialMatrix += self.calTangentialMatrix2(carPos, q[int(self.index_com[i][j])], 2 * self.carR,
                                                                  theta,
                                                                  sigma0, goal)
                    M = np.eye(3) + repulsiveMatrix + tangentialMatrix
                    T_ = self.calculateT(q[int(self.index_com[i][j])], carPos, 2 * self.carR)
                    vp = np.exp(-T_ / self.lam) * v[int(self.index_com[i][j])]
                elif int(self.index_com[i][j]) >= self.numberofcar:
                    repulsiveMatrix += self.calRepulsiveMatrix2(carPos,
                                                                obsq[int(self.index_com[i][j] - self.numberofcar)],
                                                                self.carR + self.obsR, row0, goal)
                    tangentialMatrix += self.calTangentialMatrix2(carPos,
                                                                  obsq[int(self.index_com[i][j] - self.numberofcar)],
                                                                  self.carR + self.obsR, theta,
                                                                  sigma0, goal)
                    M = np.eye(3) + repulsiveMatrix + tangentialMatrix

                    T_ = self.calculateT2(obsq[int(self.index_com[i][j] - self.numberofcar)], carPos,
                                          self.carR + self.obsR)
                    vp = np.exp(-T_ / self.lam) * obsv[int(self.index_com[i][j] - self.numberofcar)]
                ubar += (M.dot(u - vp.reshape(-1, 1)).T + vp.reshape(1, -1)).squeeze()

            # 限制ubar的模长，避免进入障碍内部后轨迹突变
            if self.calVecLen(ubar) > 5:
                ubar = ubar / self.calVecLen(ubar) * 5
            if qBefore[0] is None:
                carNextPos = carPos + ubar * self.stepSize
            else:
                carNextPos = carPos + ubar * self.stepSize
                # _, _, _, _, carNextPos = self.kinematicConstrant(carPos, qBefore, carNextPos)
            carNextPos[2] = carPos[2]

            for j in range(len(obsq)):
                if self.distanceCost(carNextPos, obsq[j]) < (self.obsR + self.carR):

                    point1 = [carNextPos[0] + (carNextPos[1] - carPos[1]) * 1000,
                              carNextPos[1] - (carNextPos[0] - carPos[0]) * 1000]
                    point2 = [carNextPos[0] - (carNextPos[1] - carPos[1]) * 1000,
                              carNextPos[1] + (carNextPos[0] - carPos[0]) * 1000]
                    cross_pos = self.line_intersect_circle((obsq[j][0], obsq[j][1], self.obsR + self.carR + 0.1),
                                                           (point1[0], point1[1]), (point2[0], point2[1]))

                    try:
                        if (self.distanceCost(np.array([cross_pos[0][0], cross_pos[0][1]]),
                                              np.array([carPos[0], carPos[1]]))
                                < self.distanceCost(np.array([cross_pos[1][0], cross_pos[1][1]]),
                                                    np.array([carPos[0], carPos[1]]))):
                            carNextPos[0] = cross_pos[0][0]
                            carNextPos[1] = cross_pos[0][1]
                        else:
                            carNextPos[0] = cross_pos[1][0]
                            carNextPos[1] = cross_pos[1][1]
                    except Exception as e:
                        carNextPos = obsq[j] + (self.obsR + self.carR + 0.1) * (carNextPos - obsq[j]) / np.linalg.norm(
                            carNextPos - obsq[j])

            for j in range(len(q)):
                if j != i:
                    if self.distanceCost(carNextPos, q[j]) < (2 * self.carR):
                        point1 = [carNextPos[0] + (carNextPos[1] - carPos[1]) * 1000,
                                  carNextPos[1] - (carNextPos[0] - carPos[0]) * 1000]
                        point2 = [carNextPos[0] - (carNextPos[1] - carPos[1]) * 1000,
                                  carNextPos[1] + (carNextPos[0] - carPos[0]) * 1000]
                        cross_pos = self.line_intersect_circle((q[j][0], q[j][1], 2 * self.carR + 0.2),
                                                               (point1[0], point1[1]), (point2[0], point2[1]))
                        try:
                            if (self.distanceCost(np.array([cross_pos[0][0], cross_pos[0][1]]),
                                                  np.array([carPos[0], carPos[1]]))
                                    < self.distanceCost(np.array([cross_pos[1][0], cross_pos[1][1]]),
                                                        np.array([carPos[0], carPos[1]]))):
                                carNextPos[0] = cross_pos[0][0]
                                carNextPos[1] = cross_pos[0][1]
                            else:
                                carNextPos[0] = cross_pos[1][0]
                                carNextPos[1] = cross_pos[1][1]
                        except Exception as e:
                            carNextPos = q[j] + (2 * self.carR + 0.1) * (carNextPos - q[j]) / np.linalg.norm(
                                carNextPos - q[j])

            if carNextPos[0] < - self.x_max:
                carNextPos[0] = - self.x_max + 0.5
            if carNextPos[0] > self.x_max:
                carNextPos[0] = self.x_max - 0.5
            if carNextPos[1] < - self.y_max:
                carNextPos[1] = - self.y_max + 0.5
            if carNextPos[1] > self.y_max:
                carNextPos[1] = self.y_max - 0.5

            carNextVel = (carNextPos - carPos) / np.linalg.norm(carNextPos - carPos) * self.V1
            vNext.append(carNextVel)
        return vNext

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(
                np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2

        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    def line_intersect_circle(self, p, lsp, esp):  # 计算直线和圆的交点
        # p is the circle parameter, lsp and lep is the two end of the line
        x0, y0, r0 = p
        x1, y1 = lsp
        x2, y2 = esp
        x0 = round(x0, 2)
        y0 = round(y0, 2)
        r0 = round(r0, 2)
        x1 = round(x1, 2)
        y1 = round(y1, 2)
        x2 = round(x2, 2)
        y2 = round(y2, 2)

        if r0 == 0:
            return [[x1, y1]]
        if x1 == x2:
            if abs(r0) >= abs(x1 - x0):
                p1 = x1, round(y0 - math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
                p2 = x1, round(y0 + math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
                inp = [p1, p2]
                # select the points lie on the line segment
                inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
            else:
                inp = []
        else:
            k = (y1 - y2) / (x1 - x2)
            b0 = y1 - k * x1
            a = k ** 2 + 1
            b = 2 * k * (b0 - y0) - 2 * x0
            c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                p1x = round((-b - math.sqrt(delta)) / (2 * a), 5)
                p2x = round((-b + math.sqrt(delta)) / (2 * a), 5)
                p1y = round(k * p1x + b0, 5)
                p2y = round(k * p2x + b0, 5)
                inp = [[p1x, p1y], [p2x, p2y]]
                # select the points lie on the line segment
                inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
            else:
                inp = []
        return inp if inp != [] else [[x1, y1]]

    @staticmethod
    def distanceCost(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def get_vertical_vector(self, vec):
        """ 求二维的向量的垂直向量 """
        assert isinstance(vec, list) and len(vec) == 2, r'平面上的向量必须为2'
        return [vec[1], -vec[0]]

    def initField(self, pos, V0, goal):
        """计算初始流场，返回列向量。"""
        temp1 = pos[0] - goal[0]
        temp2 = pos[1] - goal[1]
        temp3 = pos[2] - goal[2]
        temp4 = self.distanceCost(pos, goal)
        return -np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * V0 / temp4

    @staticmethod
    def partialDerivativeSphere(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = pos[2] - obs[2]
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 2) / r ** 2

    @staticmethod
    def partialDerivativeSphere2(obs, pos, r):
        """计算球障碍物方程偏导数，返回列向量。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return np.array([temp1, temp2, temp3], dtype=float).reshape(-1, 1) * 2 / r ** 2

    @staticmethod
    def calculateT2(obs, pos, r):
        """计算T。"""
        temp1 = pos[0] - obs[0]
        temp2 = pos[1] - obs[1]
        temp3 = 0
        return (temp1 ** 2 + temp2 ** 2 + temp3 ** 3) / r ** 2

    def calPathLen(self, path):
        """计算一个轨迹的长度。"""
        num = path.shape[0]
        len = 0
        for i in range(num - 1):
            len += self.distanceCost(path[i, :], path[i + 1, :])
        return len

    def trans(self, originalPoint, xNew, yNew, zNew):
        """
        坐标变换后地球坐标下坐标
        newX, newY, newZ是新坐标下三个轴上的方向向量
        返回列向量
        """
        lenx = self.calVecLen(xNew)
        cosa1 = xNew[0] / lenx
        cosb1 = xNew[1] / lenx
        cosc1 = xNew[2] / lenx

        leny = self.calVecLen(yNew)
        cosa2 = yNew[0] / leny
        cosb2 = yNew[1] / leny
        cosc2 = yNew[2] / leny

        lenz = self.calVecLen(zNew)
        cosa3 = zNew[0] / lenz
        cosb3 = zNew[1] / lenz
        cosc3 = zNew[2] / lenz

        B = np.array([[cosa1, cosb1, cosc1],
                      [cosa2, cosb2, cosc2],
                      [cosa3, cosb3, cosc3]], dtype=float)

        invB = np.linalg.inv(B)
        return np.dot(invB, originalPoint.T)

    @staticmethod
    def calVecLen(vec):
        """计算向量模长。"""
        return np.sqrt(np.sum(vec ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp, -1, 1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    def clear_folder(self, folder_path):
        """清空文件夹中的所有内容"""
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            except Exception as e:
                print(f"无法删除 {file_path}。原因: {e}")



