import numpy as np
import random
from copy import deepcopy

from physical_law_tool import calculate_obstacles_corner, collision_response

from physical_law_tool import calculate_deceleration_motion, calculate_acceleration_motion_vector, \
    calculate_circular_motion_vector

from physical_law_tool import resolve_overlaps_vector

from typing import Union


class PhysicalLaw:
    # 不变的常量
    GRAVITY = 10  # 重力加速度
    BINS = 10  # 分箱数量

    def __init__(self,
                 cars_mass, cars_force, cars_power, cars_friction_coefficient, cars_size, cars_wheel_spacing,
                 cars_wheel_radius,
                 obstacles_center, obstacles_radius,
                 timestep,
                 collision_coefficient=0.001,
                 ):
        '''
        cars_mass: 小车质量, shape=(n,)
        cars_force: 小车的最大受力, shape=(n,)
        cars_power: 小车功率, shape=(n,)
        cars_friction_coefficient: 小车的摩擦系数, shape=(n,)
        cars_size: 小车的长宽, shape=(n, 2)
        cars_wheel_spacing: 小车的轮间距, shape=(n,)
        obstacles_corner: 障碍物的四个角的坐标, shape=(n, 4, 2)
        timestep: 时间步长
        collision_coefficient: 碰撞系数
        '''
        self.cars_mass = np.array(cars_mass)
        self.cars_force = np.array(cars_force)
        self.cars_power = np.array(cars_power)
        self.cars_friction_coefficient = np.array(cars_friction_coefficient)
        self.cars_size = np.array(cars_size)
        self.cars_wheel_spacing = np.array(cars_wheel_spacing)
        self.cars_wheel_radius = np.array(cars_wheel_radius)
        self.obstacles_corner = np.array(calculate_obstacles_corner(obstacles_center, obstacles_radius))
        self.timestep = timestep
        self.collision_coefficient = collision_coefficient

        # 车的最大受力
        self.cars_friction_force_max = self.cars_friction_coefficient * self.cars_mass * self.GRAVITY
        self.cars_force = np.minimum(self.cars_force, self.cars_friction_force_max)
        # 车的阻力（滚动摩擦力）
        self.cars_friction_force_rolling = self.cars_friction_coefficient / self.cars_wheel_radius * self.cars_mass * self.GRAVITY

    def get_qvNext(self, q, v, vNext, dead_index) -> Union[list, list]:
        '''
        从位置q, 速度v, 下一时刻的目标速度vNext, 以及小车自身属性和障碍物设置, 计算下一时刻的位置qNext
        todo: 用numpy加速
        未考虑因素：
        1. 车的转动惯性
        2. 车的电机扭矩
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        vNext: 下一时刻的目标速度, shape=(n, 2)
        dead_index: 死亡的小车的索引, shape=(n,)
        return: 下一时刻的位置qNext, shape=(n, 3)
        return: 下一时刻的速度vNext, shape=(n, 2)
        '''
        q = np.array(deepcopy(q), dtype=np.float64)
        v = np.array(deepcopy(v), dtype=np.float64)
        vNext = np.array(vNext)
        dead_index = dead_index == 1
        mask_v = np.linalg.norm(v, axis=1) < 1e-100
        mask_vNext = np.linalg.norm(vNext, axis=1) < 1e-100
        v[mask_v] = vNext[mask_v] * 1e-100
        vNext[mask_vNext] = v[mask_vNext] * 1e-100
        mask = mask_v & mask_vNext
        v_ones = np.ones(v.shape)
        v_ones[:, 2] = 0
        v[mask] = v_ones[mask] * 1e-100
        vNext[mask] = v_ones[mask] * 1e-100
        # # 如果aAcural的模长为0，则将v乘1e-10赋值给vActual
        # v[:, :2] = vActual
        # return qActual, v
        qActual, vActual = self.get_qvNext_formula(q, v[:, :2], vNext[:, :2], dead_index)
        q[:, :2] = qActual
        v[:, :2] = vActual
        v[dead_index] = 0
        return q, v

    def get_qvNext_formula(self, q, v, vNext, dead_index) -> Union[list, list]:
        '''
        从位置q, 速度v, 下一时刻的目标速度vNext, 以及小车自身属性和障碍物设置, 使用公式法计算下一时刻的位置qNext
        q: 位置, z轴数据固定, shape=(n, 3)
        v: 速度, shape=(n, 2)
        vNext: 下一时刻的目标速度, shape=(n, 2)
        dead_index: 死亡的小车的索引, shape=(n,)
        return: 下一时刻的位置qNext, shape=(n, 3)
        return: 下一时刻的速度vNext, shape=(n, 2)
        '''
        t_last = np.ones(self.cars_mass.shape) * self.timestep
        t_last[dead_index] = 0
        q_2d = q[:, :2].copy()
        v_2d = v.copy()

        # 判断需要加速还是减速
        accelerate_condition = np.linalg.norm(vNext, axis=1) > np.linalg.norm(v, axis=1)
        # 如果需要加速则分配全部时间给转向
        t_turn = t_last * accelerate_condition
        # 计算能提供的转向加速度
        a = self.cars_friction_force_max / self.cars_mass
        # 转向
        v_2d, q_2d, t = calculate_circular_motion_vector(q_2d, v_2d, vNext, a, t_turn)
        t_last -= t
        # 加速
        v_2d, q_2d, t = calculate_acceleration_motion_vector(self.cars_mass, q_2d, v_2d, vNext, self.cars_power,
                                                             self.cars_friction_force_max,
                                                             self.cars_friction_force_rolling, t_last)
        t_last -= t

        # 判断是否需要减速
        decelerate_condition = np.linalg.norm(vNext, axis=1) < np.linalg.norm(v, axis=1)
        # 如果需要减速则分配全部时间给减速
        t_turn = t_last * decelerate_condition
        # 减速
        v_2d, q_2d, t = calculate_deceleration_motion(self.cars_mass, q_2d, v_2d, vNext, self.cars_friction_force_max,
                                                      self.cars_friction_force_rolling, t_turn)
        t_last -= t

        # 全体转向
        v_2d, q_2d, t = calculate_circular_motion_vector(q_2d, v_2d, vNext, a, t_last)
        t_last -= t
        # 剩余时间匀速运动
        q_2d += v_2d * t_last[:, np.newaxis]

        # 碰撞检测
        q_2d, v_2d = self.check_collisions(q_2d, v_2d)

        return q_2d, v_2d

    def get_cars_corners(self, centers, velocities):
        """
        根据所有小车的中心点位置、长宽和速度方向，计算所有小车的四个角标（向量化版本）
        :param centers: 所有小车的中心点位置，形状为 (m, 2)
        :param velocities: 所有小车的速度，形状为 (m, 2)
        :return: 所有小车的四个角标，形状为 (m, 4, 2)
        """
        m = centers.shape[0]
        car_sizes = self.cars_size  # 假设 self.cars_size 形状为 (m, 2)

        # 计算半宽和半高
        half_widths = car_sizes[:, 0] / 2  # (m,)
        half_heights = car_sizes[:, 1] / 2  # (m,)

        # 生成基础角点模板（未缩放）
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)  # (4, 2)

        # 通过广播生成所有小车的角点（缩放后）
        corners = np.empty((m, 4, 2))
        corners[:, :, 0] = half_widths[:, np.newaxis] * base_corners[:, 0]  # x分量
        corners[:, :, 1] = half_heights[:, np.newaxis] * base_corners[:, 1]  # y分量

        # 计算旋转角度
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])  # (m,)

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # 构建旋转矩阵数组 (m, 2, 2)
        rotation_matrices = np.empty((m, 2, 2))
        rotation_matrices[:, 0, 0] = cos_angles
        rotation_matrices[:, 0, 1] = -sin_angles
        rotation_matrices[:, 1, 0] = sin_angles
        rotation_matrices[:, 1, 1] = cos_angles

        # 应用旋转：等价于 corners[i] @ rotation_matrices[i].T
        rotated_corners = np.einsum('...ij,...kj->...ik', corners, rotation_matrices)

        # 平移角点到中心点
        cars_corners = rotated_corners + centers[:, np.newaxis, :]

        return cars_corners

    def check_collisions(self, car_centers, car_velocities, collision_time=0.01):
        """
        检查小车与障碍物以及小车与小车之间是否发生碰撞，并计算碰撞后小车的位置和速度
        :param obstacles: 所有障碍物的四个角标，形状为 (n, 4, 2)
        :param car_centers: 所有小车的中心点位置，形状为 (m, 2)
        :param car_velocities: 所有小车的速度，形状为 (m, 2)
        :param collision_time: 碰撞时间
        :return: 一个包含四个元素的元组，分别为：
                 1. 碰撞后所有小车的位置，形状为 (m, 2)
                 2. 碰撞后所有小车的速度，形状为 (m, 2)
        """
        new_car_centers = car_centers.copy()
        new_car_velocities = car_velocities.copy()
        cars_corners = self.get_cars_corners(car_centers, car_velocities)
        # 处理重叠的情况
        cars_corners, new_car_velocities = collision_response(cars_corners, new_car_velocities, self.cars_mass,
                                                              self.cars_size, self.obstacles_corner,
                                                              self.collision_coefficient, self.timestep / 10)
        cars_corners = resolve_overlaps_vector(cars_corners, self.obstacles_corner)
        new_car_centers = np.average(cars_corners, axis=1)
        return new_car_centers, new_car_velocities
