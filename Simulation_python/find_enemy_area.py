import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull, QhullError


class FindEnemyArea:
    def __init__(self, history_data, obstacles_pos, delta_time, obstacle_size, X_range, Y_range):
        """
        Initialize the trajectory predictor with initial conditions.

        Parameters:
            history_positions (dict of list of numpy.array): Historical positions of multiple vehicles.
            delta_time (float): Time interval between each step.
        """

        self.delta_time = delta_time
        self.positions = {vehicle_id: positions for vehicle_id, positions in history_data}
        self.velocities = {vehicle_id: self.calculate_velocities(vehicle_id) for vehicle_id in self.positions}

        self.history_data = history_data

        self.obstacles = []
        for i in range(len(obstacles_pos)):
            self.obstacles.append(np.array([obstacles_pos[i][0], obstacles_pos[i][1]]))

        self.obstacles_size = obstacle_size

        self.x_max, self.y_max = X_range, Y_range
    def calculate_velocities(self, vehicle_id):
        """
        Calculate velocities based on position differences.
        """
        positions = self.positions[vehicle_id]
        velocities = []
        for i in range(1, len(positions)):
            velocity = (positions[i] - positions[i - 1]) / self.delta_time
            velocities.append(velocity)
        velocities.append(velocities[-1] if velocities else np.array([0, 0]))  # Append last known velocity
        return velocities

    def is_collision(self, position):
        # 处理obstacles_size是单一值的情况
        if isinstance(self.obstacles_size, (list, tuple)):
            sizes = self.obstacles_size
        else:
            sizes = [self.obstacles_size] * len(self.obstacles)  # 创建与障碍物数量相等的尺寸列表

        for obstacle_center, size in zip(self.obstacles, sizes):
            if np.linalg.norm(position - obstacle_center) < size:
                return True
        return False

    def turn_direction(self, position, velocity, obstacle_center):
        # Calculate vector from position to obstacle
        to_obstacle = obstacle_center - position
        # Calculate vector perpendicular to velocity
        perpendicular = np.array([-velocity[1], velocity[0]])
        # Determine direction to turn based on dot product
        if np.dot(perpendicular, to_obstacle) > 0:
            # Turn right
            return np.array([velocity[1], -velocity[0]])
        else:
            # Turn left
            return np.array([-velocity[1], velocity[0]])

    def avoid_collision(self, position, velocity):
        if not isinstance(self.obstacles_size, (list, tuple)):
            sizes = [self.obstacles_size] * len(self.obstacles)
        else:
            sizes = self.obstacles_size

        for obstacle_center, size in zip(self.obstacles, sizes):
            if np.linalg.norm(position - obstacle_center) < size:
                # Calculate new direction
                return self.turn_direction(position, velocity, obstacle_center)
        return velocity

    def predict_trajectory(self, num_steps=6):
        predictions = {}
        for vehicle_id in self.positions:
            current_acceleration = self.calculate_acceleration(vehicle_id)
            current_position = np.array(self.positions[vehicle_id][-1], dtype=float)
            current_velocity = np.array(self.velocities[vehicle_id][-1], dtype=float)

            trajectory = [current_position.copy()]
            for _ in range(num_steps):
                current_velocity += current_acceleration * self.delta_time
                new_position = current_position + current_velocity * self.delta_time + 0.5 * current_acceleration * (
                            self.delta_time ** 2)
                # 检查并修正位置约束
                new_position = self.apply_boundaries(new_position)

                if self.is_collision(new_position):
                    current_velocity = self.avoid_collision(current_position, current_velocity)
                    new_position = current_position + current_velocity * self.delta_time + 0.5 * current_acceleration * (
                            self.delta_time ** 2)
                    # 再次应用边界约束
                    new_position = self.apply_boundaries(new_position)

                trajectory.append(new_position.copy())
                current_position = new_position
            predictions[vehicle_id] = np.array(trajectory)
        return predictions


    def apply_boundaries(self, position):
        """
        Apply boundaries to keep the position within the specified limits.
        """
        x_min, x_max = -self.x_max, self.x_max
        y_min, y_max = -self.y_max, self.y_max
        position[0] = max(x_min, min(x_max, position[0]))
        position[1] = max(y_min, min(y_max, position[1]))
        return position


    def calculate_acceleration(self, vehicle_id):
        """
        Calculate acceleration based on velocity differences.
        """
        velocities = self.velocities[vehicle_id]
        if len(velocities) < 2:
            return np.array([0, 0])  # Default acceleration for insufficient data
        velocity_previous = velocities[-2]
        velocity_current = velocities[-1]
        return (velocity_current - velocity_previous) / self.delta_time


    def is_inside_obstacle(self, position):
        for obstacle_center in self.obstacles:
            if np.linalg.norm(position - obstacle_center) < self.obstacles_size:
                return True
        return False

    def potential_field_adjustment(self, position, step_size=0.1, max_iter=1000):
        """ Adjust position using potential fields to escape obstacles. """
        for _ in range(max_iter):
            gradient = np.zeros_like(position)
            for center in self.obstacles:
                vector_to_center = position - center
                distance_to_center = np.linalg.norm(vector_to_center)
                if distance_to_center < self.obstacles_size:
                    force_direction = vector_to_center / distance_to_center
                    force_magnitude = self.obstacles_size - distance_to_center
                    gradient += force_direction * force_magnitude

            if np.linalg.norm(gradient) < 1e-5:
                break  # 如果梯度太小，说明已处于最优位置

            # 沿梯度方向更新位置
            position += step_size * gradient / np.linalg.norm(gradient)

        return position

    def adjust_position(self, position):
        """ 更复杂的调整策略以避免仍然位于障碍物内的情况 """
        adjusted_position = self.potential_field_adjustment(position)
        return adjusted_position


    def find_nearest_center(self, predict_trajectories, my_location,
                                use_optimal_threshold=True,
                                min_threshold=5, max_threshold=200, step=5):
        all_points = np.vstack(list(predict_trajectories.values()))
        if len(predict_trajectories) == 1:
            # If only one trajectory, use the last point of this trajectory as the "nearest center"
            single_trajectory = list(predict_trajectories.values())[0]
            nearest_center = single_trajectory[-1]
            if self.is_inside_obstacle(nearest_center):
                nearest_center = self.adjust_position(nearest_center)
            # nearest_distance = np.linalg.norm(nearest_center - my_location)
            return nearest_center

        if use_optimal_threshold:
            best_score = -1
            best_threshold = None
            for threshold in np.arange(min_threshold, max_threshold, step):
                cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='ward',
                                                  metric='euclidean')
                labels = cluster.fit_predict(all_points)
                if len(np.unique(labels)) > 1:  # Only calculate silhouette score if more than one cluster is formed
                    score = silhouette_score(all_points, labels)
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
        # distance_threshold = find_optimal_threshold(all_points) if use_optimal_threshold else 40
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=best_threshold, metric='euclidean',
                                          linkage='ward')
        labels = cluster.fit_predict(all_points)

        # Compute cluster centers
        centers = []
        for label in np.unique(labels):
            cluster_points = all_points[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)

        # Find the nearest center
        distances = [np.linalg.norm(center - my_location) for center in centers]
        nearest_index = np.argmin(distances)
        nearest_center = centers[nearest_index]

        if self.is_inside_obstacle(nearest_center):
            nearest_center = self.adjust_position(nearest_center)

        return nearest_center
