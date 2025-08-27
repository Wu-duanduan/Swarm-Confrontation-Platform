import numpy as np
from random import randint
import pygame
import rendering
import rendering2


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        # 标识
        self.id = None
        self.enemy = None
        self.size = 0.2
        self.color = None
        # 状态
        self.pos = np.zeros(2)
        self.speed = 0  # 速度标量
        self.yaw = 0  # 偏航角，0为x轴正方向，逆时针为正，(-pi, pi)
        # 约束
        # 对抗相关
        self.death = False
        # 感知范围和攻击范围
        self.attack_range = 0
        self.attack_angle = 0
        self.sensor_range_l = 0
        self.sensor_range_w = 0
        self.sensor_angle = 0


class Battle(object):
    """为避免神经网络输入数值过大，采用等比例缩小模型"""

    def __init__(self, args, X_range, Y_range, obsCenter, obsR):
        super(Battle, self).__init__()
        self.args = args
        self.dt = 1  # simulation interval，1 second
        self.t = 0
        self.render_geoms = None
        self.render_geoms_xform = None
        self.num_CARs = args.num_RCARs + args.num_BCARs
        self.num_cars = args.num_Bcars
        self.num_RCARs = args.num_RCARs
        self.num_BCARs = args.num_BCARs
        self.num_Bcars = args.num_Bcars
        self.CARs = [Model(args) for _ in range(self.num_CARs)]
        self.cars = [Model(args) for _ in range(self.num_cars)]
        self.RCARs = []
        self.BCARs = []
        self.Bcars = []
        for i, CAR in enumerate(self.CARs):
            CAR.id = i
            if i < args.num_BCARs:
                CAR.enemy = False
                CAR.color = np.array([0, 0, 1])
                CAR.attack_range = args.attack_range_B
                CAR.attack_angle = args.attack_angle_BR
                self.BCARs.append(CAR)
            elif i < args.num_RCARs + args.num_BCARs:
                CAR.enemy = True
                CAR.color = np.array([1, 0, 0])
                CAR.attack_range = args.attack_range_R
                CAR.attack_angle = args.attack_angle_BR
                self.RCARs.append(CAR)
        for i, car in enumerate(self.cars):
            car.id = i
            car.enemy = False
            car.color = np.array([0, 0, 1])
            car.sensor_range_l = args.sensor_range_B_l
            car.sensor_range_w = args.sensor_range_B_w
            car.sensor_angle = args.sensor_angle_B
            self.Bcars.append(car)
        self.sensor_range_l = args.sensor_range_B_l
        self.sensor_range_w = args.sensor_range_B_w
        self.viewer = None
        self.action_space = []
        self.reset()
        self.x_max, self.y_max = X_range, Y_range
        self.obstacle = obsCenter
        self.obsR = obsR

    def reset(self):
        self.t = 0
        # reset render
        self.render_geoms = None
        self.render_geoms_xform = None
        random_side = randint(0, 1)
        for i, CAR in enumerate(self.CARs):
            CAR.being_attacked = False
            CAR.death = False
        for i, car in enumerate(self.cars):
            car.being_attacked = False
            car.death = False
            # if not car.enemy:
            #     interval = 2.0 / (self.num_Rcars + 1)
            #     car.pos = np.array([random_side * 1.8 - 0.9, 1 - (i + 1) * interval])
            #     car.yaw = pi * random_side
            # else:
            #     interval = 2.0 / (self.num_Bcars + 1)
            #     car.pos = np.array([(1 - random_side) * 1.8 - 0.9, 1 - (i - self.num_Rcars + 1) * interval])
            #     car.yaw = pi * (1 - random_side)

    def render(self, pos, vel, fire_car, flag_car, HP_index, HP_num, missle_index, missle_num, mode='rgb_array'):
        pos_copy = np.copy(pos)
        vel_copy = np.copy(vel)

        if self.viewer is None:
            self.viewer = rendering.Viewer(900, 480)
            pygame.init()
        # 每次渲染时清除旧的几何对象
        self.render_geoms = []
        self.render_geoms_xform = []
        # 初始化pygame用于文本渲染

        for i, CAR in enumerate(self.CARs):  # 添加无人车以及攻击范围
            if flag_car[i] == 1:
                CAR.color = np.array([0, 0, 0])
            xform = rendering.Transform()
            for x in rendering.make_CAR(CAR.size):
                x.set_color(*CAR.color, 0.5)
                x.add_attr(xform)
                self.render_geoms.append(x)
                self.render_geoms_xform.append(xform)

            start = pos_copy[i][0:2]  # 箭头的起点为车辆当前位置
            end = pos_copy[i][0:2] + vel_copy[i][0:2] / np.linalg.norm(vel_copy[i][0:2]) * 0.5  # 箭头的终点为目标位置

            # 绘制箭头线段
            arrow_line = rendering.Line(start, end, dashed=False, linewidth=4)
            # arrow_line = self.draw_dashed_line(start, end, 0.5)
            arrow_line.set_color(*CAR.color, 0.5)
            self.render_geoms.append(arrow_line)
            self.render_geoms_xform.append(xform)

            sector = rendering.make_circle(radius=0.1)
            sector.set_color(*CAR.color, 0.8)
            sector.add_attr(xform)
            self.render_geoms.append(sector)
            self.render_geoms_xform.append(xform)

        self.length_temp1 = len(self.render_geoms)

        for i, CAR in enumerate(self.CARs):  # 添加无人车以及攻击范围
            xform = rendering.Transform()
            if fire_car[i] >= 0:
                start = pos_copy[i][0:2]  # 箭头的起点为车辆当前位置
                end = pos_copy[int(fire_car[i])][0:2]  # 箭头的终点为目标位置
                # 绘制箭头线段
                arrow_line = rendering.Line(start, end, dashed=True, linewidth=4)
                # arrow_line = self.draw_dashed_line(start, end, 0.5)
                arrow_line.set_color(*CAR.color, 1)
                self.render_geoms.append(arrow_line)
                self.render_geoms_xform.append(xform)

            # 动态绘制血条
            health_bar_width = 0.5  # 每个格子的宽度
            health_bar_height = 0.1  # 格子的高度
            max_health = HP_num  # 假设血量最大为 100
            num_cells = HP_num  # 假设血条由 10 个格子组成

            # 创建血条的位置变换
            health_xform = rendering.Transform()

            # 动态绘制血条格子
            for j in range(num_cells):
                # 每个格子的位置
                x_offset = j * health_bar_width - 0.5  # 水平偏移，使得格子居中
                health_bar = rendering.FilledPolygon([
                    (x_offset, 0),  # 左下角
                    (x_offset + health_bar_width, 0),  # 右下角
                    (x_offset + health_bar_width, health_bar_height),  # 右上角
                    (x_offset, health_bar_height)  # 左上角
                ])
                if HP_index[i] == 0:
                    health_bar.set_color(1, 1, 1, 0)  # 白色表示无血量
                elif j < HP_index[i] / (max_health / num_cells):  # 计算应该显示的格子数量
                    health_bar.set_color(255/255, 165/255, 0)  # 红色表示有血量
                else:
                    health_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无血量

                if HP_index[i] != 0:
                    # 为血条设置位置和旋转
                    health_bar.add_attr(health_xform)
                    # 设置血条的 Y 坐标偏移，使其在 car 上方
                    health_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + CAR.size + 0.5)  # 偏移 0.3 让血条在 car 上方
                self.render_geoms.append(health_bar)
                self.render_geoms_xform.append(xform)

            # 动态绘制弹药
            missle_bar_width = 0.5  # 每个格子的宽度
            missle_bar_height = 0.1  # 格子的高度
            max_missle = missle_num  # 假设弹药最大为 100
            num_cells = int(missle_num)  # 假设弹药由 10 个格子组成

            # 创建弹药的位置变换
            missle_xform = rendering.Transform()

            # 动态绘制弹药格子
            for j in range(num_cells):
                # 每个格子的位置
                x_offset = j * missle_bar_width - 0.5  # 水平偏移，使得格子居中
                missle_bar = rendering.FilledPolygon([
                    (x_offset, 0),  # 左下角
                    (x_offset + missle_bar_width, 0),  # 右下角
                    (x_offset + missle_bar_width, missle_bar_height),  # 右上角
                    (x_offset, missle_bar_height)  # 左上角
                ])

                if missle_index[i] == 0 or HP_index[i] == 0:
                    missle_bar.set_color(1, 1, 1, 0)  # 白色表示无弹药
                elif j < missle_index[i] / (max_missle / num_cells):  # 计算应该显示的格子数量
                    missle_bar.set_color(0, 1, 0)  # 绿色表示有弹药
                else:
                    missle_bar.set_color(0.5, 0.5, 0.5)  # 灰色表示无弹药
                if missle_index[i] != 0:
                    # 为弹药设置位置和旋转
                    missle_bar.add_attr(missle_xform)
                    # 设置弹药的 Y 坐标偏移，使其在 car 上方
                    missle_xform.set_translation(pos_copy[i][0] - 0.5, pos_copy[i][1] + CAR.size + 0.3)  # 偏移 0.3 让弹药在 car 上方
                self.render_geoms.append(missle_bar)
                self.render_geoms_xform.append(xform)

        # 渲染静态障碍物
        self.render_static_obstacles()
        # self.render_static_obstacles2(self.obstacle)
        self.viewer.geoms = []
        for geom in self.render_geoms:
            self.viewer.add_geom(geom)

        self.viewer.set_bounds(-self.x_max, self.x_max, -self.y_max, self.y_max)

        for i, CAR in enumerate(self.CARs):  # 无人车以及攻击范围需要旋转
            idx_ratio = self.length_temp1 // self.num_CARs
            for idx in range(idx_ratio):
                self.render_geoms_xform[idx_ratio * i + idx].set_translation(*pos_copy[i][0:2])

                if vel_copy[i][1] >= 0 and vel_copy[i][0] >= 0:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                elif vel_copy[i][1] < 0 and vel_copy[i][0] >= 0:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]))
                else:
                    self.render_geoms_xform[idx_ratio * i + idx].set_rotation(
                        np.arctan(vel_copy[i][1] / vel_copy[i][0]) + np.pi)

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_arrow(self, pos, llm_goal, flag_car, mode='rgb_array'):
        pos_copy = np.copy(pos)
        llm_goal_copy = np.copy(llm_goal)
        # 添加箭头表示速度方向
        for i, CAR in enumerate(self.CARs):
            if i < self.num_CARs / 2 and flag_car[i] == 0:
                # 计算箭头的起点和终点
                start = pos_copy[i][0:2]  # 箭头的起点为车辆当前位置
                end = llm_goal_copy[i][0:2]  # 箭头的终点为目标位置

                # 绘制箭头线段
                arrow_line = rendering.Line(start, end, dashed=True)
                # arrow_line = self.draw_dashed_line(start, end, 0.5)
                arrow_line.set_color(0, 0, 1)
                self.render_geoms.append(arrow_line)

                # 绘制箭头头部（三角形）
                # arrow_head = rendering.make_polygon([(0, 0), (-0.2, 0.1), (-0.2, -0.1)])  # 小三角形
                # arrow_head.set_color(1, 0, 0)  # 红色
                # arrow_head_xform = rendering.Transform()
                # arrow_head_xform.set_translation(*end)  # 将箭头头部移动到终点
                # arrow_head_xform.set_rotation(np.arctan2(llm_goal_copy[i][1], llm_goal_copy[i][0]))  # 旋转箭头头部
                # arrow_head.add_attr(arrow_head_xform)
                # self.render_geoms.append(arrow_head)

        for geom in self.render_geoms:
            self.viewer.add_geom(geom)

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def vel2yaw(self, vel):
        if vel[1] >= 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        elif vel[1] < 0 and vel[0] >= 0:
            return np.arctan(vel[1] / vel[0])
        else:
            return np.arctan(vel[1] / vel[0]) + np.pi

    def render_static_obstacles(self, ego_pos=(0, 0), ego_yaw=0, BEV_mode=False):
        # 定义静态障碍物：位置 (x, y)、宽度、高度、基础颜色
        obstacles = [
            # 左侧垂直障碍物 (x ≈ -8.95)
            {"pos": (-8.95, -4.55), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},  # 稍深的混凝土灰
            {"pos": (-8.95, -2.45), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.95, 1.95), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.95, 4.15), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},

            # 左侧第二排垂直障碍物 (x ≈ -4.95)
            {"pos": (-4.95, -4.55), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, -2.45), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, 1.95), "width": 0.2, "height": 0.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-4.95, 4.15), "width": 0.2, "height": 1.0, "color": (0.4, 0.4, 0.4)},

            # 左侧水平连接障碍物 (y ≈ -4.55 和 y ≈ -1.65)
            {"pos": (-8.75, -4.55), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, -1.65), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, 1.95), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-8.75, 4.95), "width": 2.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 左侧小水平连接块
            {"pos": (-5.45, -4.55), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, -1.65), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, 1.95), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-5.45, 4.95), "width": 0.5, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 右侧垂直障碍物 (x ≈ 4.75)
            {"pos": (4.75, -5.55), "width": 0.2, "height": 1.6, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, -2.65), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, 0.85), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.75, 3.85), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},

            # 右侧第二排垂直障碍物 (x ≈ 8.75)
            {"pos": (8.75, -5.55), "width": 0.2, "height": 1.6, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, -2.65), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, 0.85), "width": 0.2, "height": 1.7, "color": (0.4, 0.4, 0.4)},
            {"pos": (8.75, 3.85), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},

            # 右侧水平连接障碍物 (y ≈ -5.55 和 y ≈ -1.15)
            {"pos": (4.95, -5.55), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, -1.15), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, 0.85), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (4.95, 5.45), "width": 1.1, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 右侧第二排水平连接障碍物
            {"pos": (7.35, -5.55), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, -1.15), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, 0.85), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (7.35, 5.45), "width": 1.4, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 中间区域障碍物
            {"pos": (-1.15, -3.55), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (-1.15, 1.75), "width": 0.2, "height": 1.9, "color": (0.4, 0.4, 0.4)},
            {"pos": (-0.95, -3.55), "width": 1.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (-0.95, 3.45), "width": 1.0, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 中间右侧小水平障碍物
            {"pos": (1.95, -3.55), "width": 0.7, "height": 0.2, "color": (0.4, 0.4, 0.4)},
            {"pos": (1.95, 3.45), "width": 0.7, "height": 0.2, "color": (0.4, 0.4, 0.4)},

            # 中间右侧垂直障碍物
            {"pos": (2.65, -3.55), "width": 0.2, "height": 1.8, "color": (0.4, 0.4, 0.4)},
            {"pos": (2.65, 1.75), "width": 0.2, "height": 1.9, "color": (0.4, 0.4, 0.4)},
        ]

        for obs in obstacles:
            xform = rendering2.Transform()

            # 非 BEV 模式下应用透视变形，增强 3D 效果
            if not BEV_mode:
                skew_factor = 0.75  # 顶部宽度为底部的 75%，更强透视感
                rect_points = [
                    (0, 0),
                    (obs["width"], 0),
                    (obs["width"] * skew_factor, obs["height"]),
                    (0, obs["height"]),
                ]
            else:
                rect_points = [
                    (0, 0),
                    (obs["width"], 0),
                    (obs["width"], obs["height"]),
                    (0, obs["height"]),
                ]

            # 创建阴影（先渲染，确保在障碍物下方）
            shadow_offset = 0.1  # 减小阴影偏移量，使阴影更贴近
            shadow = rendering2.make_polygon([
                (0, 0),
                (obs["width"], 0),
                (obs["width"], obs["height"]),
                (0, obs["height"]),
            ], filled=True)
            shadow.set_color(0.1, 0.1, 0.1, 0.75)  # 稍柔和的深色阴影
            shadow_xform = rendering2.Transform()
            shadow_xform.set_translation(obs["pos"][0] + shadow_offset, obs["pos"][1] - shadow_offset)
            shadow.add_attr(shadow_xform)
            self.render_geoms.append(shadow)
            self.render_geoms_xform.append(shadow_xform)

            # 创建主障碍物矩形
            rect = rendering2.make_polygon(rect_points, filled=True)
            rect.set_color(*obs["color"], 0.9)  # 稍深的混凝土灰，微透明
            rect.add_attr(xform)
            self.render_geoms.append(rect)
            self.render_geoms_xform.append(xform)
            xform.set_translation(*obs["pos"])

            # 添加更粗的斜面，增强立体感
            bevel_size = 0.2  # 保持较大的斜面尺寸
            # 顶部斜面（高光）
            top_bevel = rendering2.make_polygon([
                (0, obs["height"]),
                (obs["width"], obs["height"]),
                (obs["width"] - bevel_size, obs["height"] - bevel_size),
                (bevel_size, obs["height"] - bevel_size),
            ], filled=True)
            top_bevel.set_color(1.0, 1.0, 1.0, 0.9)  # 纯白高光
            top_bevel.add_attr(xform)
            self.render_geoms.append(top_bevel)

            # 左侧斜面（高光）
            left_bevel = rendering2.make_polygon([
                (0, 0),
                (bevel_size, bevel_size),
                (bevel_size, obs["height"] - bevel_size),
                (0, obs["height"]),
            ], filled=True)
            left_bevel.set_color(1.0, 1.0, 1.0, 0.9)
            left_bevel.add_attr(xform)
            self.render_geoms.append(left_bevel)

            # 底部斜面（阴影）
            bottom_bevel = rendering2.make_polygon([
                (0, 0),
                (obs["width"], 0),
                (obs["width"] - bevel_size, bevel_size),
                (bevel_size, bevel_size),
            ], filled=True)
            bottom_bevel.set_color(0.1, 0.1, 0.1, 0.9)  # 深黑阴影
            bottom_bevel.add_attr(xform)
            self.render_geoms.append(bottom_bevel)

            # 右侧斜面（阴影）
            right_bevel = rendering2.make_polygon([
                (obs["width"], 0),
                (obs["width"], obs["height"]),
                (obs["width"] - bevel_size, obs["height"] - bevel_size),
                (obs["width"] - bevel_size, bevel_size),
            ], filled=True)
            right_bevel.set_color(0.1, 0.1, 0.1, 0.9)
            right_bevel.add_attr(xform)
            self.render_geoms.append(right_bevel)

            # BEV 模式下添加顶部盖子，突出高度
            if BEV_mode:
                cap_offset = 0.4  # 保持较大的盖子偏移量
                cap = rendering2.make_polygon([
                    (0, obs["height"]),
                    (obs["width"], obs["height"]),
                    (obs["width"], obs["height"] + cap_offset),
                    (0, obs["height"] + cap_offset),
                ], filled=True)
                cap.set_color(1.0, 1.0, 1.0, 0.9)  # 纯白顶部，模拟强光
                cap_xform = rendering2.Transform()
                cap_xform.set_translation(obs["pos"][0], obs["pos"][1])
                cap.add_attr(cap_xform)
                self.render_geoms.append(cap)
                self.render_geoms_xform.append(cap_xform)

    def close(self):
        """Close the viewer properly."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
