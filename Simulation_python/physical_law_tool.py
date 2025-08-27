########################################
# 用来为动力学约束提供工具函数
########################################
import numpy as np


#################################################################################
# 计算障碍物的四个角的坐标，仅在初始化时调用一次，暂时不进行优化
#################################################################################
def calculate_obstacles_corner(obstacles_center, obstacles_radius=0.1) -> list:
    '''
    从障碍物的中心点和半径, 计算出障碍物的四个角的坐标
    obstacles_center: 障碍物的中心点, shape=(n, 2)
    obstacles_radius: 障碍物的半径, shape=1
    return: 障碍物的四个角的坐标, shape=(n, 4, 2)
    '''
    if len(obstacles_center) == 0:
        return []
    obstacles_center = np.array(obstacles_center)
    return calculate_all_rectangles(obstacles_center[:, :2], obstacles_radius * 2)


def is_connected(center1, center2, side_length):
    """判断两个小障碍物是否连接"""
    x1, y1 = center1
    x2, y2 = center2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx <= side_length and dy == 0) or (dy <= side_length and dx == 0)


def dfs(centers, index, visited, group, side_length):
    """深度优先搜索，将相互连接的小障碍物加入同一组"""
    visited[index] = True
    group.append(centers[index])
    for i in range(len(centers)):
        if not visited[i] and is_connected(centers[index], centers[i], side_length):
            dfs(centers, i, visited, group, side_length)


def group_connected_obstacles(centers, side_length):
    """将小障碍物分组"""
    visited = [False] * len(centers)
    groups = []
    for i in range(len(centers)):
        if not visited[i]:
            group = []
            dfs(centers, i, visited, group, side_length)
            groups.append(group)
    return groups


def calculate_rectangle_corners(group, side_length):
    """计算一组小障碍物组成的长方形的四角坐标"""
    group = np.array(group)
    x_coords = group[:, 0]
    y_coords = group[:, 1]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    half_side = side_length / 2
    bottom_left = (min_x - half_side, min_y - half_side)
    bottom_right = (max_x + half_side, min_y - half_side)
    top_right = (max_x + half_side, max_y + half_side)
    top_left = (min_x - half_side, max_y + half_side)
    return [bottom_left, bottom_right, top_right, top_left]


def calculate_all_rectangles(centers, side_length=0.1):
    """计算所有小障碍物分组后组成的长方形的四角坐标"""
    groups = group_connected_obstacles(centers, side_length)
    rectangles = []
    for group in groups:
        corners = calculate_rectangle_corners(group, side_length)
        rectangles.append(corners)
    return rectangles


def calculate_rectangle_size(rectangle):
    """根据长方形的四个角标计算长方形的大小"""
    x_coords = [corner[0] for corner in rectangle]
    y_coords = [corner[1] for corner in rectangle]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width, height


#########################################################################
# 检测两个物体相撞后的位置和速度
#########################################################################
def collision_response(c, v, m, size, obstacles, collision_coefficient=0.8, timestep=0.01):
    '''
    检测物体相撞后的位置和速度，物体均是轴对称的（长方形）
    n: 小车的数量
    m: 障碍物的数量
    c: 小车的四个角标, shape=(n,4,2)
    v: 小车的速度, shape=(n,2,)
    m: 小车的质量, shape=(n,)
    obstacles: 障碍物的四个角标, shape=(m,4,2)
    collision_coefficient: 碰撞系数, shape=1
    timestep: 时间步长, shape=1
    return: 小车相撞后的位置, shape=(n,4,2)
    return: 小车相撞后的速度, shape=(n,2,)
    '''
    # 先用包围盒算法粗略检测是否重叠
    # 先计算包围盒
    num_cars = c.shape[0]
    num_obstacles = obstacles.shape[0]

    # 向量化计算小车包围盒
    car_min_x = np.min(c[:, :, 0], axis=1)
    car_max_x = np.max(c[:, :, 0], axis=1)
    car_min_y = np.min(c[:, :, 1], axis=1)
    car_max_y = np.max(c[:, :, 1], axis=1)

    # 向量化计算障碍物包围盒
    if len(obstacles) > 0:
        obs_min_x = np.min(obstacles[:, :, 0], axis=1)
        obs_max_x = np.max(obstacles[:, :, 0], axis=1)
        obs_min_y = np.min(obstacles[:, :, 1], axis=1)
        obs_max_y = np.max(obstacles[:, :, 1], axis=1)

        #################################################
        # 小车与障碍物
        #################################################
        # 广播以检查所有小车和障碍物的组合
        car_min_x = car_min_x[:, np.newaxis]
        car_max_x = car_max_x[:, np.newaxis]
        car_min_y = car_min_y[:, np.newaxis]
        car_max_y = car_max_y[:, np.newaxis]

        # 计算是否重叠
        overlaps_x = (car_min_x < obs_max_x) & (car_max_x > obs_min_x)
        overlaps_y = (car_min_y < obs_max_y) & (car_max_y > obs_min_y)
        overlaps = overlaps_x & overlaps_y

        for i in range(num_cars):
            for j in range(num_obstacles):
                if overlaps[i, j]:
                    car = c[i]
                    obstacle = obstacles[j]
                    # 再用分离轴定理检测是否重叠
                    result = sat_collision(car, obstacle)
                    if result is not None:
                        # 再分离重叠的矩形
                        # c[i] = separate_overlap(car, obstacle)
                        insert_points1, insert_points2 = result
                        (c[i], v[i]), _ = collision_separation(c[i], v[i], m[i], size[i], insert_points1, obstacles[j],
                                                               np.zeros_like(v[i]), np.inf,
                                                               calculate_rectangle_size(obstacles[j]), insert_points2,
                                                               collision_coefficient, timestep)

    ##################################################################
    # 小车与小车
    ##################################################################
    # 重新计算小车包围盒用于小车之间的检测
    car_min_x = np.min(c[:, :, 0], axis=1)
    car_max_x = np.max(c[:, :, 0], axis=1)
    car_min_y = np.min(c[:, :, 1], axis=1)
    car_max_y = np.max(c[:, :, 1], axis=1)

    # 广播以检查所有小车和小车的组合
    car_min_x_1 = car_min_x[:, np.newaxis]
    car_max_x_1 = car_max_x[:, np.newaxis]
    car_min_y_1 = car_min_y[:, np.newaxis]
    car_max_y_1 = car_max_y[:, np.newaxis]

    overlaps_x_car = (car_min_x_1 < car_max_x) & (car_max_x_1 > car_min_x)
    overlaps_y_car = (car_min_y_1 < car_max_y) & (car_max_y_1 > car_min_y)
    overlaps_car = overlaps_x_car & overlaps_y_car

    for i in range(num_cars):
        for j in range(i + 1, num_cars):  # 避免重复检查
            if overlaps_car[i, j]:
                car1 = c[i]
                car2 = c[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car1, car2)
                if result is not None:
                    # 再分离重叠的矩形
                    # c[i], c[j] = separate_overlap_car(car1, car2)
                    insert_points1, insert_points2 = result
                    (c[i], v[i]), (c[j], v[j]) = collision_separation(c[i], v[i], m[i], size[i], insert_points1, c[j],
                                                                      v[j], m[j], size[j], insert_points2,
                                                                      collision_coefficient, timestep)

    return c, v


def collision_separation(c1, v1, m1, size1, insert_points1, c2, v2, m2, size2, insert_points2,
                         collision_coefficient=0.8, timestep=0.01):
    """
    碰撞分离，根据碰撞法线，计算接触点，从而计算两个物体碰撞后的位置和速度（需考虑旋转）

    参数:
    c1 (numpy.ndarray): 物体1的四个角标，形状为 (4, 2)。
    v1 (numpy.ndarray): 物体1的速度，形状为 (2,)。
    m1 (float): 物体1的质量。
    size1 (tuple): 物体1的大小，形状为 (2,)。
    insert_points1 (numpy.ndarray): 物体1的插入点，形状为 (4,)，类型是 bool。
    c2 (numpy.ndarray): 物体2的四个角标，形状为 (4, 2)。
    v2 (numpy.ndarray): 物体2的速度，形状为 (2,)。
    m2 (float): 物体2的质量。
    size2 (tuple): 物体2的大小，形状为 (2,)。
    insert_points2 (numpy.ndarray): 物体2的插入点，形状为 (4,)，类型是 bool。
    collision_coefficient (float, 可选): 碰撞系数，默认为 0.8。
    timestep (float, 可选): 时间步长，默认为 0.01。

    返回:
    tuple: 包含物体1和物体2碰撞后的位置和速度的元组。
    """

    # 旋转
    def calculate_rotation_angle(corner, center, m, size, insert_points, v, timestep, collision_coefficient):
        # 计算插入点中点
        insert_point = np.mean(insert_points, axis=0)

        # 计算相对位置向量（从质心到碰撞点）
        r_vector = insert_point - center

        # 计算转动惯量
        width = size[0]
        height = size[1]
        I = m * (width ** 2 + height ** 2) / 12  # 矩形转动惯量公式

        # 计算扭矩（r × F，这里F近似为速度方向）
        torque = np.cross(r_vector, v)

        # 计算角加速度（Δω = τ/I * timestep）
        rotation_factor = torque / (I + 1e-6) * timestep * collision_coefficient

        # 计算最终旋转角度（限制最大旋转幅度）
        return rotation_factor

    # 首先，我们需要计算两个多边形的质心
    centroid1 = np.mean(c1, axis=0)
    centroid2 = np.mean(c2, axis=0)

    # 计算旋转角度
    if m2 == np.inf:
        rotation_angle1 = 0
        rotation_angle2 = 0
    else:
        # rotation_angle1 = calculate_rotation_angle_center(centroid1, m1, size1, centroid2, v2, timestep, collision_coefficient)
        # rotation_angle2 = calculate_rotation_angle_center(centroid2, m2, size2, centroid1, v1, timestep, collision_coefficient)
        rotation_angle1 = calculate_rotation_angle(c1, centroid1, m1, size1, insert_points2, v2, timestep,
                                                   collision_coefficient)
        rotation_angle2 = calculate_rotation_angle(c2, centroid2, m2, size2, insert_points1, v1, timestep,
                                                   collision_coefficient)
    # 构建旋转矩阵
    rotation_matrix1 = np.array([
        [np.cos(rotation_angle1), -np.sin(rotation_angle1)],
        [np.sin(rotation_angle1), np.cos(rotation_angle1)]
    ])
    rotation_matrix2 = np.array([
        [np.cos(rotation_angle2), -np.sin(rotation_angle2)],
        [np.sin(rotation_angle2), np.cos(rotation_angle2)]
    ])

    # 将角标相对于质心的位置旋转
    car1_separated_relative = c1 - centroid1
    car1_separated_rotated_relative = np.dot(car1_separated_relative, rotation_matrix1)

    # 将旋转后的角标位置还原到原坐标系
    c1 = car1_separated_rotated_relative + centroid1

    if m2 != np.inf:
        car2_separated_relative = c2 - centroid2
        car2_separated_rotated_relative = np.dot(car2_separated_relative, rotation_matrix2)
        c2 = car2_separated_rotated_relative + centroid2

    # 对两个多边形的速度进行旋转
    v1 = np.dot(v1, rotation_matrix1)
    v2 = np.dot(v2, rotation_matrix2) if m2 != np.inf else v2

    # 速度减小
    v1 = v1 * collision_coefficient
    v2 = v2 * collision_coefficient

    return (c1, v1), (c2, v2)


def separate_overlap(car, other):
    """
    分离两个重叠的多边形
    """
    dx = np.mean(other[:, 0]) - np.mean(car[:, 0])
    dy = np.mean(other[:, 1]) - np.mean(car[:, 1])
    if dx != 0:
        car[:, 0] -= np.sign(dx) * 0.1
    if dy != 0:
        car[:, 1] -= np.sign(dy) * 0.1
    return car


def separate_overlap_car(car1, car2):
    """
    分离两个重叠的多边形
    两个多边形向反方向移动
    """
    # 首先，我们需要计算两个多边形的质心
    centroid1 = np.mean(car1, axis=0)
    centroid2 = np.mean(car2, axis=0)

    # 计算从质心1到质心2的向量
    separation_vector = centroid2 - centroid1

    # 归一化分离向量
    separation_vector = separation_vector / np.linalg.norm(separation_vector)

    # 假设我们每次分离移动一个固定的小距离，这里设为0.1
    separation_distance = 0.1

    # 两个多边形向反方向移动
    car1_separated = car1 - separation_vector * separation_distance
    car2_separated = car2 + separation_vector * separation_distance

    return car1_separated, car2_separated


def resolve_overlaps_vector(cars_corners, obstacles):
    """
    分离重叠的矩形
    cars_corners: 小车的四个角标, shape=(n, 4, 2)
    obstacles: 障碍物的四个角标, shape=(m, 4, 2)
    return: 分离重叠的矩形后的小车的四个角标, shape=(n, 4, 2)
    """
    # 先用包围盒算法粗略检测是否重叠
    # 先计算包围盒
    num_cars = cars_corners.shape[0]
    num_obstacles = obstacles.shape[0]

    # 向量化计算小车包围盒
    car_min_x = np.min(cars_corners[:, :, 0], axis=1)
    car_max_x = np.max(cars_corners[:, :, 0], axis=1)
    car_min_y = np.min(cars_corners[:, :, 1], axis=1)
    car_max_y = np.max(cars_corners[:, :, 1], axis=1)

    if len(obstacles) > 0:
        # 向量化计算障碍物包围盒
        obs_min_x = np.min(obstacles[:, :, 0], axis=1)
        obs_max_x = np.max(obstacles[:, :, 0], axis=1)
        obs_min_y = np.min(obstacles[:, :, 1], axis=1)
        obs_max_y = np.max(obstacles[:, :, 1], axis=1)

        #################################################
        # 小车与障碍物
        #################################################
        # 广播以检查所有小车和障碍物的组合
        car_min_x = car_min_x[:, np.newaxis]
        car_max_x = car_max_x[:, np.newaxis]
        car_min_y = car_min_y[:, np.newaxis]
        car_max_y = car_max_y[:, np.newaxis]

        # 计算是否重叠
        overlaps_x = (car_min_x < obs_max_x) & (car_max_x > obs_min_x)
        overlaps_y = (car_min_y < obs_max_y) & (car_max_y > obs_min_y)
        overlaps = overlaps_x & overlaps_y

        for i in range(num_cars):
            for j in range(num_obstacles):
                if overlaps[i, j]:
                    car = cars_corners[i]
                    obstacle = obstacles[j]
                    # 再用分离轴定理检测是否重叠
                    result = sat_collision(car, obstacle)
                    if result is not None:
                        # 再分离重叠的矩形
                        cars_corners[i] = separate_overlap(car, obstacle)

    ##################################################################
    # 小车与小车
    ##################################################################
    # 重新计算小车包围盒用于小车之间的检测
    car_min_x = np.min(cars_corners[:, :, 0], axis=1)
    car_max_x = np.max(cars_corners[:, :, 0], axis=1)
    car_min_y = np.min(cars_corners[:, :, 1], axis=1)
    car_max_y = np.max(cars_corners[:, :, 1], axis=1)

    # 广播以检查所有小车和小车的组合
    car_min_x_1 = car_min_x[:, np.newaxis]
    car_max_x_1 = car_max_x[:, np.newaxis]
    car_min_y_1 = car_min_y[:, np.newaxis]
    car_max_y_1 = car_max_y[:, np.newaxis]

    overlaps_x_car = (car_min_x_1 < car_max_x) & (car_max_x_1 > car_min_x)
    overlaps_y_car = (car_min_y_1 < car_max_y) & (car_max_y_1 > car_min_y)
    overlaps_car = overlaps_x_car & overlaps_y_car

    for i in range(num_cars):
        for j in range(i + 1, num_cars):  # 避免重复检查
            if overlaps_car[i, j]:
                car1 = cars_corners[i]
                car2 = cars_corners[j]
                # 再用分离轴定理检测是否重叠
                result = sat_collision(car1, car2)
                if result is not None:
                    # 再分离重叠的矩形
                    cars_corners[i], cars_corners[j] = separate_overlap_car(car1, car2)

    return cars_corners


###################################################################################
# 匀速圆周运动的计算
###################################################################################
def calculate_circular_motion_vector(q, v, vNext, a, timestep):
    """
    计算多个小车在一段时间内的匀速圆周运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    a (numpy.ndarray): 小车的向心加速度，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    # from log import log
    # log(q)
    # log(v)
    # log(vNext)
    # log(a)
    # log(timestep)
    # 计算初速度和目标速度的夹角
    dot_product = np.sum(v * vNext, axis=1)
    norm_v = np.linalg.norm(v, axis=1)
    norm_vNext = np.linalg.norm(vNext, axis=1)
    cos_angle = dot_product / (norm_v * norm_vNext)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    # 计算角速度
    angular_velocity = a / norm_v

    # 计算达到目标速度所需的时间
    required_time = angle / angular_velocity

    # 确定实际使用的时间
    actual_time = np.minimum(required_time, timestep)

    # 计算旋转角度
    rotation_angle = angular_velocity * actual_time

    # 确定旋转方向
    v_3d = np.pad(v, ((0, 0), (0, 1)), mode='constant')
    vNext_3d = np.pad(vNext, ((0, 0), (0, 1)), mode='constant')
    cross_product = np.cross(v_3d, vNext_3d)[:, 2]
    rotation_angle = np.where(cross_product < 0, -rotation_angle, rotation_angle)

    # 构建旋转矩阵
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ]).transpose(2, 0, 1)

    # 计算最终速度
    final_velocities = np.einsum('nij,nj->ni', rotation_matrix, v)

    # 计算圆周运动的半径
    radius = norm_v ** 2 / a

    # 计算圆心方向
    center_direction = np.stack([-v[:, 1], v[:, 0]], axis=1)
    center_direction = center_direction / np.linalg.norm(center_direction, axis=1, keepdims=True)

    # 将 cross_product < 0 扩展为形状 (n, 2)
    condition = np.stack([cross_product < 0, cross_product < 0], axis=1)

    # 根据条件调整圆心方向
    center_direction = np.where(condition, -center_direction, center_direction)

    # 计算圆心位置
    center = q + center_direction * radius[:, np.newaxis]

    # 计算初始角度
    initial_angle = np.arctan2(q[:, 1] - center[:, 1], q[:, 0] - center[:, 0])

    # 计算最终角度
    final_angle = initial_angle + rotation_angle

    # 计算最终位置
    final_positions = center + radius[:, np.newaxis] * np.stack([np.cos(final_angle), np.sin(final_angle)], axis=1)

    # 保存经过的时间
    elapsed_times = actual_time

    return final_velocities, final_positions, elapsed_times


###################################################################################
# 加速直线运动的计算（固定功率和力的最大值）
###################################################################################
def calculate_acceleration_motion_vector(mass, q, v, vNext, power, force_max, friction_force, timestep):
    """
    计算多个小车在一段时间内的加速直线运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    mass (numpy.ndarray): 小车的质量，形状为 (n,)。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    power (numpy.ndarray): 小车的功率，形状为 (n,)。
    force_max (numpy.ndarray): 小车的最大力，形状为 (n,)。
    friction_force (numpy.ndarray): 小车的摩擦力，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    n = len(mass)
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    # 计算速度大小和方向
    v_size = np.linalg.norm(v, axis=1)
    vNext_size = np.linalg.norm(vNext, axis=1)
    direction = vNext / vNext_size[:, np.newaxis]

    # 计算临界速度
    vDiv = power / force_max

    # 第一阶段：速度未达到，功率提供的动力大于force_max
    vCondition1 = (v_size < vDiv) & (elapsed_times < timestep) & (v_size < vNext_size)
    a = (force_max - friction_force) / mass
    vTarget1 = np.minimum(vDiv, vNext_size)
    required_time1 = (vTarget1 - v_size) / a
    time_use1 = np.minimum(required_time1, timestep)
    v_size[vCondition1] = v_size[vCondition1] + a[vCondition1] * time_use1[vCondition1]
    v[vCondition1] = v_size[vCondition1][:, np.newaxis] * direction[vCondition1]
    q[vCondition1] = q[vCondition1] + v[vCondition1] * time_use1[vCondition1][:, np.newaxis] + 0.5 * a[vCondition1][:,
                                                                                                     np.newaxis] * \
                     direction[vCondition1] * time_use1[vCondition1][:, np.newaxis] ** 2
    elapsed_times[vCondition1] += time_use1[vCondition1]

    # 第二阶段：速度达到临界速度，功率提供的动力小于force_max
    vCondition2 = (v_size >= vDiv) & (elapsed_times < timestep) & (v_size < vNext_size)
    v_max = power / friction_force - 1e-10
    vTarget2 = np.where(vNext_size < v_max, vNext_size, v_max)
    c1 = -mass / friction_force
    c2 = mass * power / (friction_force ** 2)
    c3 = friction_force / power
    t0 = c1 * v_size - c2 * np.log(1 - c3 * v_size)
    t1 = c1 * vTarget2 - c2 * np.log(1 - c3 * vTarget2)
    single_time_use = np.where(t1 - t0 < timestep - elapsed_times, t1 - t0, timestep - elapsed_times)
    t2 = t0 + single_time_use
    vActual = calculate_v(t2, mass, friction_force, power)
    x1 = power * single_time_use / friction_force
    x2 = mass * (vActual - v_size) * (vActual + v_size) / (2 * friction_force)
    front_size = x1 - x2
    q[vCondition2] = q[vCondition2] + direction[vCondition2] * front_size[vCondition2][:, np.newaxis]
    v[vCondition2] = vActual[vCondition2][:, np.newaxis] * direction[vCondition2]
    elapsed_times[vCondition2] += single_time_use[vCondition2]

    final_velocities = v
    final_positions = q

    return final_velocities, final_positions, elapsed_times


from scipy.special import lambertw


def calculate_v(t, m, f, P):
    """
    计算速度 v 作为时间 t 的函数
    参数:
        t : 时间（标量或数组）
        m, f, P : 物理常数（标量）
    返回:
        v : 速度值（实数部分）
    """
    exponent = - (f ** 2) / (m * P) * t
    arg = -np.exp(exponent - 1)  # -e^{exponent - 1} = - (e^{-1} * e^{exponent})
    w = lambertw(arg, k=0)  # 主分支(k=0)
    return (P / f) * (1 + w.real)  # 提取实数部分


##########################################################################################
# 匀减速直线运动的计算
##########################################################################################
def calculate_deceleration_motion(mass, q, v, vNext, force_max, friction_force, timestep):
    """
    计算多个小车在一段时间内的匀减速直线运动的最终速度、最终位置和经过的时间。

    参数:
    n (int): 小车的数量。
    mass (numpy.ndarray): 小车的质量，形状为 (n,)。
    q (numpy.ndarray): 小车的初始位置，形状为 (n, 2)。
    v (numpy.ndarray): 小车的初始速度，形状为 (n, 2)。
    vNext (numpy.ndarray): 小车的目标速度，形状为 (n, 2)。
    power (numpy.ndarray): 小车的功率，形状为 (n,)。
    force_max (numpy.ndarray): 小车的最大力，形状为 (n,)。
    friction_force (numpy.ndarray): 小车的摩擦力，形状为 (n,)。
    timestep (numpy.ndarray): 每个小车的时间步长，形状为 (n,)。

    返回:
    final_velocities (numpy.ndarray): 每个小车的最终速度，形状为 (n, 2)。
    final_positions (numpy.ndarray): 每个小车的最终位置，形状为 (n, 2)。
    elapsed_times (numpy.ndarray): 每个小车达到最终状态所经过的时间，形状为 (n,)。
    """
    n = len(mass)
    # 初始化结果数组
    final_velocities = np.zeros((n, 2))
    final_positions = np.zeros((n, 2))
    elapsed_times = np.zeros(n)

    # 计算力的大小
    force = force_max + friction_force

    # 计算速度大小和方向
    v_size = np.linalg.norm(v, axis=1)
    vNext_size = np.linalg.norm(vNext, axis=1)
    direction = -v / v_size[:, np.newaxis]

    # 计算加速度
    a = force / mass

    # 计算时间
    time_use = np.minimum(timestep, (v_size - vNext_size) / a)

    # 计算最终速度和位置
    # 使用NumPy向量化操作替代循环
    final_velocities = v + a[:, np.newaxis] * time_use[:, np.newaxis] * direction
    final_positions = q + v * time_use[:, np.newaxis] + 0.5 * a[:, np.newaxis] * direction * time_use[:,
                                                                                             np.newaxis] ** 2
    elapsed_times = time_use

    return final_velocities, final_positions, elapsed_times


#########################################################################################################
# 碰撞检测
#########################################################################################################

def sat_collision(c1, c2):
    """
    使用分离轴定理（SAT）检测两个矩形是否发生碰撞。

    参数:
    c1 (numpy.ndarray): 第一个矩形的四个角的坐标，形状为 (4, 2)。
    c2 (numpy.ndarray): 第二个矩形的四个角的坐标，形状为 (4, 2)。

    返回:
    tuple: 如果发生碰撞，返回在对方内部的点；如果未发生碰撞，返回 None。
    """
    # 存储所有可能的分离轴
    axes = []
    # 遍历第一个矩形的每条边
    for i in range(4):
        # 获取当前边的向量
        edge = c1[(i + 1) % 4] - c1[i]
        # 计算垂直于边的轴
        axis = np.array([-edge[1], edge[0]])
        # 归一化轴向量
        axis = axis / np.linalg.norm(axis)
        axes.append(axis)

    # 遍历第二个矩形的每条边
    for i in range(4):
        # 获取当前边的向量
        edge = c2[(i + 1) % 4] - c2[i]
        # 计算垂直于边的轴
        axis = np.array([-edge[1], edge[0]])
        # 归一化轴向量
        axis = axis / np.linalg.norm(axis)
        axes.append(axis)

    # 初始化在对方内部的点（全在）
    insert_points1 = np.ones(c1.shape[0], dtype=bool)
    insert_points2 = np.ones(c2.shape[0], dtype=bool)

    # 遍历所有分离轴
    for axis in axes:
        # 投影第一个矩形的顶点到当前轴上
        projections_c1 = np.dot(c1, axis)
        # 找到投影的最小值和最大值
        min_c1 = np.min(projections_c1)
        max_c1 = np.max(projections_c1)

        # 投影第二个矩形的顶点到当前轴上
        projections_c2 = np.dot(c2, axis)
        # 找到投影的最小值和最大值
        min_c2 = np.min(projections_c2)
        max_c2 = np.max(projections_c2)

        # 检查投影是否重叠
        if max_c1 < min_c2 or max_c2 < min_c1:
            # 如果没有重叠，则两个矩形没有碰撞
            return None

        # 计算那些点在对方内部
        insert_points1 = np.logical_and(insert_points1, (projections_c1 >= min_c2) & (projections_c1 <= max_c2))
        insert_points2 = np.logical_and(insert_points2, (projections_c2 >= min_c1) & (projections_c2 <= max_c1))

    return insert_points1, insert_points2
