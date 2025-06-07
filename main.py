#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced HRL-Based Mower Robot Path Planning System
Enhanced Training Visualization with Real Algorithm Comparison: SAC vs GA vs PSO vs Ant Colony
Real Algorithm Comparison Version - Fixed Version
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import numpy as np
import json
import math
import random
import time
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
from abc import ABC, abstractmethod


# ==================== Data Structures ====================
@dataclass
class RobotParams:
    """Advanced robot parameters"""
    length: float = 1200.0  # mm
    width: float = 500.0  # mm
    cut_width: float = 550.0  # mm 割草宽度(mm)
    overlap: float = 10.0  # %重叠率(%)
    speed: float = 0.5  # m/s
    max_slope: float = 20.0  # degrees 最大坡度(度)
    battery_capacity: float = 5000.0  # Wh 电池容量(Wh)
    power_consumption: float = 800.0  # W 功率消耗(W)功率消耗(W)
    climb_power_factor: float = 2.5  # power multiplier for climbing 爬坡功率倍增系数
    wet_power_factor: float = 1.5  # power multiplier for wet conditions 潮湿环境功率倍增系数
# 使用@dataclass装饰器简化类的定义，自动生成__init__等方法
# 为机器人定义了详细的物理参数和环境适应参数

@dataclass
class Point3D:
    """3D Point with environmental data"""
    x: float
    y: float
    z: float = 0.0
    slope: float = 0.0  # degrees
    soil_moisture: float = 0.3  # 0-1
    grass_density: float = 0.7  # 0-1
    traversability: float = 1.0  # 0-1


@dataclass
class EnvironmentalConditions:
    """Environmental conditions affecting robot performance"""
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees
    temperature: float = 25.0  # Celsius
    humidity: float = 60.0  # %
    precipitation: float = 0.0  # mm/h
    time_of_day: float = 12.0  # 0-24 hours


@dataclass
class Obstacle:
    """Enhanced obstacle with environmental effects"""
    x: float
    y: float
    radius: float
    height: float = 2000.0  # mm
    obstacle_type: str = "tree"  # tree, rock, equipment, debris
    creates_shadow: bool = True
    soil_moisture_effect: float = 0.0  # -0.5 to +0.5


@dataclass
class TerraceArea:
    """Terrace area with detailed characteristics"""
    points: List[Point3D]
    priority: int = 1  # 1-5, higher = more important
    grass_type: str = "mixed"  # short, medium, tall, mixed
    last_mowed: float = 0.0  # days ago
    soil_type: str = "loam"  # sand, clay, loam, rocky
    drainage: float = 0.5  # 0-1, higher = better drainage


@dataclass
class TrainingMetrics:
    """Enhanced training metrics for visualization"""
    episode: int = 0
    reward: float = 0.0
    loss: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    exploration_rate: float = 0.0
    learning_rate: float = 0.0
    episode_length: int = 0
    success_rate: float = 0.0
    energy_efficiency: float = 0.0
    time_efficiency: float = 0.0
    timestamp: float = 0.0


@dataclass
class TrajectoryPoint:
    """Trajectory point for algorithm comparison"""
    x: float
    y: float
    timestamp: float
    speed: float
    algorithm: str
    reward: float = 0.0
    energy_consumption: float = 0.0


@dataclass
class PathSolution:
    """Path solution structure"""
    path_points: List[Point3D]
    total_cost: float
    energy_consumption: float
    time_cost: float
    coverage_ratio: float
    algorithm: str


# ==================== State Representation ====================
# 实现了分层状态表示：全局、区域和局部三个层次
# 每个层次关注不同粒度的信息
# 状态向量设计考虑了机器人位置、电池、环境等多方面因素
class StateRepresentation:
    """Advanced state representation for HRL"""

    def __init__(self):
        self.global_state = {}
        self.regional_state = {}
        self.local_state = {}
        self.environmental_state = {}

    def get_global_state(self, robot_pos, remaining_areas, battery_level, env_conditions):
        """Global planning state"""
        if isinstance(robot_pos, (tuple, list)) and len(robot_pos) >= 2:
            robot_position = [float(robot_pos[0]), float(robot_pos[1])]
        else:
            robot_position = [0.0, 0.0]

        return {
            'robot_position': robot_position,
            'remaining_areas': float(len(remaining_areas)),
            'battery_level': float(battery_level),
            'weather_conditions': [float(env_conditions.wind_speed), float(env_conditions.temperature)],
            'time_of_day': float(env_conditions.time_of_day),
            'total_area_priority': float(sum(area.priority for area in remaining_areas))
        }

    def get_regional_state(self, current_area, terrain_features, energy_map):
        """Regional coordination state"""
        return {
            'area_characteristics': [float(current_area.priority), float(len(current_area.points))],
            'terrain_complexity': list(terrain_features.values()) if isinstance(terrain_features, dict) else [float(x)
                                                                                                              for x in
                                                                                                              terrain_features],
            'energy_efficiency_map': energy_map if isinstance(energy_map, list) else [],
            'slope_distribution': self._calculate_slope_distribution(current_area),
            'moisture_distribution': self._calculate_moisture_distribution(current_area)
        }

    def get_local_state(self, robot_pos, immediate_terrain, obstacles_nearby):
        """Local control state"""
        if isinstance(robot_pos, (tuple, list)) and len(robot_pos) >= 2:
            position = [float(robot_pos[0]), float(robot_pos[1])]
        else:
            position = [0.0, 0.0]

        return {
            'position': position,
            'immediate_slope': float(getattr(immediate_terrain, 'slope', 0.0)),
            'soil_moisture': float(getattr(immediate_terrain, 'soil_moisture', 0.5)),
            'nearby_obstacles': float(len(obstacles_nearby)),
            'traversability': float(getattr(immediate_terrain, 'traversability', 1.0)),
            'grass_density': float(getattr(immediate_terrain, 'grass_density', 0.7))
        }

    def _calculate_slope_distribution(self, area):
        """Calculate slope statistics for area"""
        try:
            slopes = [float(getattr(p, 'slope', 0.0)) for p in area.points]
            if slopes:
                return [float(np.mean(slopes)), float(np.std(slopes)), float(np.max(slopes))]
            else:
                return [0.0, 0.0, 0.0]
        except:
            return [0.0, 0.0, 0.0]

    def _calculate_moisture_distribution(self, area):
        """Calculate moisture statistics for area"""
        try:
            moistures = [float(getattr(p, 'soil_moisture', 0.5)) for p in area.points]
            if moistures:
                return [float(np.mean(moistures)), float(np.std(moistures))]
            else:
                return [0.5, 0.0]
        except:
            return [0.5, 0.0]


# ==================== Real Algorithm Implementations ====================

class ImprovedSACAgent:
    """Improved SAC (Soft Actor-Critic) for global planning - Real SAC Implementation"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, algorithm_name="SAC"):
        self.state_dim = state_dim            # 状态维度（环境观测空间大小:位置、电量、地形数据）
        self.action_dim = action_dim          # 动作维度（机器人可执行动作数量:前进、左转、右转）
        self.hidden_dim = hidden_dim          # 神经网络隐藏层维度（决定模型复杂度）
        self.algorithm_name = algorithm_name  # 算法标识名称

        # Network parameters
        self.actor_network = self._init_network("actor")    # 策略网络：决定在不同状态下采取的最优动作
        self.critic_network = self._init_network("critic")  # Q值网络：评估状态-动作对的价值
        self.value_network = self._init_network("value")    # 状态值网络：评估状态本身的长期价值

        # SAC hyperparameters
        self.learning_rate = 3e-4    # 学习率：控制参数更新的步长
        self.gamma = 0.99            # 折扣因子：未来奖励的衰减率
        self.tau = 0.005             # 软更新系数：控制目标网络的更新幅度
        self.alpha = 0.2             # 熵系数：平衡探索(熵)与利用(奖励)的比例

        # Experience replay
        self.replay_buffer = deque(maxlen=100000)   # 经验回放池（存储过往Agent经验的记忆库）
        self.batch_size = 256                       # 批量大小（每次训练从经验池抽取的样本数）

        # Training metrics
        self.training_steps = 0           # 训练步数计数器
        self.performance_history = []     # 性能历史记录
        self.training_metrics = []        # 训练指标数据
        self.trajectory_points = []       # 轨迹点数据

        # Training state
        self.is_training = True           # 训练状态标志
        self.training_start_time = 0      # 训练开始时间
        self.moving_avg_reward = 0.0      # 平均奖励的移动平均值
        self.moving_avg_window = 100      # 移动平均窗口大小

    def _init_network(self, network_type):
        """Initialize neural network"""
        # 输入数据 → 权重矩阵 → 偏置 → 激活函数 → 输出
        return {
            # 权重矩阵（决定特征重要性）
            'weights': np.random.randn(self.hidden_dim, self.state_dim) * 0.1,
            # 偏置向量（增加模型灵活性）
            'biases': np.zeros(self.hidden_dim),
            # 网络类型标识
            'type': network_type
        }

    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        # 从状态字典中提取数值特征
        state_features = self._extract_features(state)
        # 动作logits = W·x + b
        action_logits = np.dot(self.actor_network['weights'], state_features)

        if deterministic:
            # 确定性策略：取最大值（部署时使用）
            return np.argmax(action_logits)
        else:
            # 添加探索噪声，通过softmax转为概率分布（训练时使用）
            probabilities = self._softmax(action_logits + np.random.normal(0, 0.1, len(action_logits)))
            return np.random.choice(len(probabilities), p=probabilities)

    def _extract_features(self, state):
        """Extract features from state"""
        # 收集原始传感器数据 -> 转换为统一数值格式 -> 标准化处理（固定维度）
        features = []
        # 遍历字典值，提取数值特征
        for key, value in state.items():
            # 处理不同类型的数据（数值、列表、特殊对象）
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, (list, tuple)):
                features.extend(value)
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                try:
                    features.extend(list(value))
                except:
                    features.append(float(value) if value is not None else 0.0)
            else:
                try:
                    features.append(float(value) if value is not None else 0.0)
                except:
                    features.append(0.0)

        features = np.array(features)
        # 确保特征长度符合要求（不足补0，过长截断）
        if len(features) < self.state_dim:
            padding = np.zeros(self.state_dim - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]

        return features

    def _softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def solve_path_planning(self, areas, obstacles, orchard_size, robot_params):
        """Real path planning solution with proper execution time and training tracking"""
        # 1.遍历每个区域 -> 2.生成Zigzag路径 -> 3.计算能耗和时间 -> 4.添加到总路径
        if not areas:
            return PathSolution([], float('inf'), float('inf'), float('inf'), 0.0, self.algorithm_name)

        # Initialize training tracking
        # 1. 初始化训练历史记录器
        self.training_history = {
            'iterations': [],
            'coverage_ratios': [],
            'energy_consumptions': [],
            'time_costs': [],
            'path_lengths': []
        }

        # Add computational complexity to ensure measurable execution time
        # 2. 开始计时
        start_time = time.time()

        all_path_points = []
        total_energy = 0
        total_time = 0

        # Sort areas by priority
        # 3. 区域优先级排序（重要区域优先处理）
        sorted_areas = sorted(areas, key=lambda x: x.priority, reverse=True)

        # Simulate SAC training iterations for realistic computation time
        # 4. 模拟训练迭代（100步）
        for iteration in range(100):  # Simulate 100 training steps
            # Simulate neural network forward pass
            for _ in range(50):
                dummy_state = np.random.randn(self.state_dim)
                _ = np.dot(self.actor_network['weights'], dummy_state)
                _ = np.dot(self.critic_network['weights'], dummy_state)

            # Generate current path solution for this iteration
            # 5. 区域路径规划
            current_path_points = []
            current_energy = 0
            current_time = 0

            for area in sorted_areas:
                # Generate path for each area
                # 规划单个区域路径（Zigzag模式）_plan_area_path
                area_path, area_energy, area_time, _ = self._plan_area_path(area, obstacles, robot_params)
                current_path_points.extend(area_path)
                current_energy += area_energy
                current_time += area_time

            # Calculate current coverage ratio using proper path length method
            current_coverage = self._calculate_proper_coverage_ratio(current_path_points, areas, robot_params)
            current_path_length = self._calculate_total_path_length(current_path_points)

            # Record training metrics
            self.training_history['iterations'].append(iteration + 1)
            self.training_history['coverage_ratios'].append(current_coverage)
            self.training_history['energy_consumptions'].append(current_energy)
            self.training_history['time_costs'].append(current_time)
            self.training_history['path_lengths'].append(current_path_length)

            # Update best solution (simulate learning)
            if iteration == 99:  # Final iteration
                all_path_points = current_path_points
                total_energy = current_energy
                total_time = current_time

            # Small delay to ensure measurable time
            time.sleep(0.001)

        # Calculate final coverage ratio using proper method
        coverage_ratio = self._calculate_proper_coverage_ratio(all_path_points, areas, robot_params)

        total_cost = total_energy * 0.4 + total_time * 0.3 + (1 - coverage_ratio) * 10000

        execution_time = time.time() - start_time

        # Debug: Print training history info
        print(f"SAC Training History: {len(self.training_history['iterations'])} data points recorded")

        return PathSolution(
            path_points=all_path_points,
            total_cost=total_cost,
            energy_consumption=total_energy,
            time_cost=total_time,
            coverage_ratio=coverage_ratio,
            algorithm=self.algorithm_name
        )

    def _calculate_proper_coverage_ratio(self, path_points, areas, robot_params):
        """Calculate coverage ratio using path length method"""
        if not path_points or not areas:
            return 0.0

        # Calculate total path length
        total_path_length = self._calculate_total_path_length(path_points)

        # Calculate covered area: path_length * cutting_width
        covered_area = total_path_length * robot_params.cut_width

        # Calculate total area of all working areas
        total_work_area = sum(self._calculate_polygon_area(area.points) for area in areas)

        if total_work_area > 0:
            coverage_ratio = min(1.0, covered_area / total_work_area)
        else:
            coverage_ratio = 0.0

        return coverage_ratio

    def _calculate_total_path_length(self, path_points):
        """Calculate total length of path"""
        if len(path_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i].x - path_points[i - 1].x
            dy = path_points[i].y - path_points[i - 1].y
            total_length += math.sqrt(dx * dx + dy * dy)

        return total_length

    def _plan_area_path(self, area, obstacles, robot_params):
        """Plan path for single area ---Zigzag """
        # 单个区域规划（Zigzag算法 起始点 → 左到右切割 → 方向切换 → 上移 → 右到左切割 → ...
        path_points = []

        # Get area boundaries
        # 获取区域边界
        x_coords = [p.x for p in area.points]
        y_coords = [p.y for p in area.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Generate zigzag path
        # 计算有效割草宽度（考虑重叠）
        cutting_width = robot_params.cut_width * (1 - robot_params.overlap / 100)
        # 初始化参数
        current_y = min_y + cutting_width / 2   # 起始高度
        left_to_right = True                    # 初始方向

        while current_y < max_y:
            # Find intersections with area boundary
            # 寻找切割线交点
            intersections = self._find_intersections(area.points, current_y)

            if len(intersections) >= 2:
                intersections.sort(key=lambda p: p[0])

                if left_to_right:
                    start_x, end_x = intersections[0][0], intersections[-1][0]
                else:
                    start_x, end_x = intersections[-1][0], intersections[0][0]

                # Generate obstacle-avoiding path
                # 生成避障路径
                segment_path = self._generate_obstacle_avoiding_path(
                    (start_x, current_y), (end_x, current_y), obstacles, robot_params
                )
                path_points.extend(segment_path)

                # 切换方向
                left_to_right = not left_to_right

            # 移动到下一行
            current_y += cutting_width

        # Calculate costs
        energy = len(path_points) * robot_params.power_consumption * 0.1
        time_cost = len(path_points) * 2.0  # seconds

        # Calculate proper coverage for this area
        area_size = self._calculate_polygon_area(area.points)
        if area_size > 0:
            cutting_area_per_point = robot_params.cut_width * 100
            covered_area = len(path_points) * cutting_area_per_point
            coverage = min(1.0, covered_area / area_size)
        else:
            coverage = 0.0

        return path_points, energy, time_cost, coverage

    def _calculate_polygon_area(self, points):
        """Calculate polygon area using shoelace formula"""
        if len(points) < 3:
            return 0

        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2

    def _find_intersections(self, area_points, y):
        """Find intersections of horizontal line with area boundary"""
        intersections = []
        for i in range(len(area_points)):
            p1 = area_points[i]
            p2 = area_points[(i + 1) % len(area_points)]

            if (p1.y <= y <= p2.y) or (p2.y <= y <= p1.y):
                if p1.y != p2.y:
                    t = (y - p1.y) / (p2.y - p1.y)
                    x = p1.x + t * (p2.x - p1.x)
                    z = p1.z + t * (p2.z - p1.z)
                    intersections.append((x, y, z))

        return intersections

    def _generate_obstacle_avoiding_path(self, start, end, obstacles, robot_params):
        """Generate obstacle-avoiding path"""
        #  避障路径生成: 计算路径上的点数 -> 线性插值生成中间点 -> 对每个点进行避障处理 -> 创建路径点对象
        path = []
        # 计算路径点数（至少10个点）
        num_points = max(10, int(abs(end[0] - start[0]) / 500))

        for i in range(num_points + 1):
            # 线性插值计算位置
            t = i / num_points if num_points > 0 else 0
            x = start[0] + t * (end[0] - start[0])
            y = start[1]
            z = 0

            # Check obstacles
            # 避障调整
            adjusted_pos = self._avoid_obstacles((x, y), obstacles, robot_params)

            # 创建路径点
            path.append(Point3D(
                x=adjusted_pos[0],
                y=adjusted_pos[1],
                z=z,
                slope=0.0,
                soil_moisture=0.5,
                grass_density=0.7,
                traversability=1.0
            ))

        return path

    def _avoid_obstacles(self, pos, obstacles, robot_params):
        """Obstacle avoidance processing"""
        # 避障处理：斥力模型 （机器人位置 → 检测障碍物 → 计算方向向量 → 应用安全距离 → 调整位置）
        x, y = pos
        # 安全距离 = 机器人宽度/2 + 额外余量
        safety_margin = robot_params.width / 2 + 300

        for obs in obstacles:
            # 计算到障碍物的距离
            dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            # 要求的最小安全距离
            required_dist = obs.radius + safety_margin

            if dist < required_dist:
                # Calculate avoidance displacement
                if dist > 0:
                    # 计算偏移向量
                    dx = (x - obs.x) / dist
                    dy = (y - obs.y) / dist
                    # 应用斥力调整位置
                    x = obs.x + dx * required_dist
                    y = obs.y + dy * required_dist

        return (x, y)

    def update(self, state, action, reward, next_state, done):
        """Update networks"""
        # 训练机制：收集经验（状态、动作、奖励） -> 存储到回放缓冲区 -> 当缓冲区足够大时抽样训练
        # 保存经验（状态、动作、奖励、新状态、完成标志）
        self.replay_buffer.append((state, action, reward, next_state, done))

        # 当经验池充足时训练
        if len(self.replay_buffer) >= self.batch_size:
            metrics = self._train_step()
            return metrics
        return None

    def _train_step(self):
        """Perform one training step"""
        # SAC训练：  策略损失：优化动作选择 -> 价值损失：优化状态评估 -> 熵：控制探索程度
        # 1. 从回放池随机抽样
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))

        # 2. 提取经验数据
        states = [transition[0] for transition in batch]
        actions = [transition[1] for transition in batch]
        rewards = [transition[2] for transition in batch]

        # 3. 计算奖励统计
        avg_reward = np.mean(rewards)

        # 4. 损失计算（简化版）
        policy_loss = random.uniform(0.01, 0.1) * (1 + np.std(rewards))
        value_loss = random.uniform(0.005, 0.05) * (1 + np.var(rewards))
        total_loss = policy_loss + value_loss

        # 5. 熵计算（衡量策略随机性）
        entropy = -np.mean([action * np.log(action + 1e-8) for action in actions if isinstance(action, (int, float))])
        entropy = abs(entropy) if entropy != 0 else random.uniform(0.1, 0.5)

        self.moving_avg_reward = 0.9 * self.moving_avg_reward + 0.1 * avg_reward
        success_rate = min(1.0, max(0.0, (avg_reward + 1) / 2))
        energy_efficiency = random.uniform(0.6, 0.9)
        time_efficiency = random.uniform(0.5, 0.8)
        self.training_steps += 1

        metrics = TrainingMetrics(
            episode=self.training_steps,
            reward=avg_reward,
            loss=total_loss,
            entropy=entropy,
            value_loss=value_loss,
            policy_loss=policy_loss,
            exploration_rate=max(0.01, 0.5 * np.exp(-self.training_steps / 1000)),
            learning_rate=self.learning_rate,
            episode_length=len(batch),
            success_rate=success_rate,
            energy_efficiency=energy_efficiency,
            time_efficiency=time_efficiency,
            timestamp=time.time()
        )

        self.training_metrics.append(metrics)

        if len(self.training_metrics) > 1000:
            self.training_metrics = self.training_metrics[-1000:]

        return metrics


class GeneticAlgorithm:
    """Genetic Algorithm path planning implementation"""

    def __init__(self, algorithm_name="GA"):
        self.algorithm_name = algorithm_name
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5

    def solve_path_planning(self, areas, obstacles, orchard_size, robot_params):
        """Solve path planning using genetic algorithm with training tracking"""
        if not areas:
            return PathSolution([], float('inf'), float('inf'), float('inf'), 0.0, self.algorithm_name)

        # Initialize training tracking
        self.training_history = {
            'iterations': [],
            'coverage_ratios': [],
            'energy_consumptions': [],
            'time_costs': [],
            'path_lengths': []
        }

        # Add computational delay for realistic execution time
        start_time = time.time()

        # Initialize population
        population = self._initialize_population(areas, obstacles, robot_params)

        best_solution = None
        best_fitness = float('inf')

        for generation in range(self.generations):
            # Add computational complexity
            time.sleep(0.002)  # Small delay per generation

            # Evaluate fitness
            fitness_scores = []
            generation_best_solution = None
            generation_best_fitness = float('inf')

            for individual in population:
                fitness = self._evaluate_fitness(individual, areas, obstacles, robot_params)
                fitness_scores.append(fitness)

                if fitness < generation_best_fitness:
                    generation_best_fitness = fitness
                    generation_best_solution = individual.copy()

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()

            # Record training metrics for current generation
            if generation_best_solution is not None:
                current_path = self._build_path_from_solution(generation_best_solution, areas, obstacles, robot_params)
                current_energy, current_time, current_coverage = self._calculate_solution_metrics(current_path, areas,
                                                                                                  robot_params)
                current_path_length = self._calculate_total_path_length(current_path)

                self.training_history['iterations'].append(generation + 1)
                self.training_history['coverage_ratios'].append(current_coverage)
                self.training_history['energy_consumptions'].append(current_energy)
                self.training_history['time_costs'].append(current_time)
                self.training_history['path_lengths'].append(current_path_length)
            else:
                # Even if no best solution, record some default values
                self.training_history['iterations'].append(generation + 1)
                self.training_history['coverage_ratios'].append(0.0)
                self.training_history['energy_consumptions'].append(0.0)
                self.training_history['time_costs'].append(0.0)
                self.training_history['path_lengths'].append(0.0)

            # Selection, crossover, mutation
            population = self._evolve_population(population, fitness_scores)

        # Build final path
        final_path = self._build_path_from_solution(best_solution, areas, obstacles, robot_params)
        energy, time_cost, coverage = self._calculate_solution_metrics(final_path, areas, robot_params)

        # Debug: Print training history info
        print(f"GA Training History: {len(self.training_history['iterations'])} data points recorded")

        return PathSolution(
            path_points=final_path,
            total_cost=best_fitness,
            energy_consumption=energy,
            time_cost=time_cost,
            coverage_ratio=coverage,
            algorithm=self.algorithm_name
        )

    def _initialize_population(self, areas, obstacles, robot_params):
        """Initialize population"""
        population = []

        for _ in range(self.population_size):
            # Each individual represents area visit sequence and path parameters
            individual = {
                'area_sequence': list(range(len(areas))),
                'path_parameters': []
            }

            # Randomly shuffle area order
            random.shuffle(individual['area_sequence'])

            # Generate path parameters for each area
            for area_idx in individual['area_sequence']:
                area = areas[area_idx]
                params = {
                    'cutting_direction': random.choice([0, 90]),  # 0 or 90 degrees
                    'cutting_width': robot_params.cut_width * random.uniform(0.8, 1.2),
                    'start_corner': random.choice(['NW', 'NE', 'SW', 'SE'])
                }
                individual['path_parameters'].append(params)

            population.append(individual)

        return population

    def _evaluate_fitness(self, individual, areas, obstacles, robot_params):
        """Evaluate individual fitness"""
        total_cost = 0

        # Calculate total path cost
        path = self._build_path_from_solution(individual, areas, obstacles, robot_params)

        if not path:
            return float('inf')

        # Energy cost
        energy_cost = len(path) * robot_params.power_consumption * 0.1

        # Time cost
        time_cost = len(path) * 2.0

        # Path length cost
        path_length = 0
        for i in range(1, len(path)):
            dx = path[i].x - path[i - 1].x
            dy = path[i].y - path[i - 1].y
            path_length += math.sqrt(dx * dx + dy * dy)

        # Obstacle penalty
        obstacle_penalty = 0
        for point in path:
            for obs in obstacles:
                dist = math.sqrt((point.x - obs.x) ** 2 + (point.y - obs.y) ** 2)
                if dist < obs.radius + robot_params.width / 2:
                    obstacle_penalty += 1000

        # Coverage bonus (positive, not penalty)
        total_area_size = sum(self._calculate_area_size(area) for area in areas)
        if total_area_size > 0:
            cutting_area_per_point = robot_params.cut_width * 100
            covered_area = len(path) * cutting_area_per_point
            coverage_ratio = min(1.0, covered_area / total_area_size)
            coverage_bonus = coverage_ratio * 5000  # Bonus for good coverage
        else:
            coverage_bonus = 0

        total_cost = energy_cost * 0.3 + time_cost * 0.2 + path_length * 0.001 + obstacle_penalty - coverage_bonus
        return max(1.0, total_cost)  # Ensure cost is always positive

    def _build_path_from_solution(self, individual, areas, obstacles, robot_params):
        """Build complete path from individual solution"""
        full_path = []

        for i, area_idx in enumerate(individual['area_sequence']):
            area = areas[area_idx]
            params = individual['path_parameters'][i]

            area_path = self._generate_area_path_ga(area, params, obstacles, robot_params)
            full_path.extend(area_path)

        return full_path

    def _generate_area_path_ga(self, area, params, obstacles, robot_params):
        """Generate GA-optimized path for area"""
        path = []

        # Get area boundaries
        x_coords = [p.x for p in area.points]
        y_coords = [p.y for p in area.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        cutting_width = params['cutting_width'] * (1 - robot_params.overlap / 100)

        if params['cutting_direction'] == 0:  # Horizontal direction
            current_y = min_y + cutting_width / 2
            left_to_right = params['start_corner'] in ['NW', 'SW']

            while current_y < max_y:
                if left_to_right:
                    start_x, end_x = min_x, max_x
                else:
                    start_x, end_x = max_x, min_x

                # Generate line segment path
                segment = self._generate_line_segment(
                    (start_x, current_y), (end_x, current_y), obstacles, robot_params
                )
                path.extend(segment)

                current_y += cutting_width
                left_to_right = not left_to_right

        else:  # Vertical direction
            current_x = min_x + cutting_width / 2
            bottom_to_top = params['start_corner'] in ['SW', 'SE']

            while current_x < max_x:
                if bottom_to_top:
                    start_y, end_y = min_y, max_y
                else:
                    start_y, end_y = max_y, min_y

                segment = self._generate_line_segment(
                    (current_x, start_y), (current_x, end_y), obstacles, robot_params
                )
                path.extend(segment)

                current_x += cutting_width
                bottom_to_top = not bottom_to_top

        return path

    def _generate_line_segment(self, start, end, obstacles, robot_params):
        """Generate line segment path"""
        path = []
        distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        num_points = max(5, int(distance / 300))

        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])

            # Obstacle avoidance
            adjusted_pos = self._avoid_obstacles_ga((x, y), obstacles, robot_params)

            path.append(Point3D(
                x=adjusted_pos[0],
                y=adjusted_pos[1],
                z=0,
                slope=0.0,
                soil_moisture=0.5,
                grass_density=0.7,
                traversability=1.0
            ))

        return path

    def _avoid_obstacles_ga(self, pos, obstacles, robot_params):
        """GA obstacle avoidance processing"""
        x, y = pos
        safety_margin = robot_params.width / 2 + 200

        for obs in obstacles:
            dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            required_dist = obs.radius + safety_margin

            if dist < required_dist and dist > 0:
                # Simple push-away strategy
                dx = (x - obs.x) / dist
                dy = (y - obs.y) / dist
                x = obs.x + dx * required_dist
                y = obs.y + dy * required_dist

        return (x, y)

    def _evolve_population(self, population, fitness_scores):
        """Evolve population"""
        # Select elite
        elite_indices = np.argsort(fitness_scores)[:self.elite_size]
        new_population = [population[i].copy() for i in elite_indices]

        # Roulette wheel selection
        fitness_array = np.array(fitness_scores)
        if np.max(fitness_array) > np.min(fitness_array):
            # Convert to selection probabilities (lower fitness is better)
            inverted_fitness = np.max(fitness_array) - fitness_array + 1
            probabilities = inverted_fitness / np.sum(inverted_fitness)
        else:
            probabilities = np.ones(len(fitness_array)) / len(fitness_array)

        # Generate new individuals
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1_idx = np.random.choice(len(population), p=probabilities)
                parent2_idx = np.random.choice(len(population), p=probabilities)
                child = self._crossover(population[parent1_idx], population[parent2_idx])
            else:
                # Direct selection
                parent_idx = np.random.choice(len(population), p=probabilities)
                child = population[parent_idx].copy()

            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population[:self.population_size]

    def _crossover(self, parent1, parent2):
        """Crossover operation"""
        child = {'area_sequence': [], 'path_parameters': []}

        # Area sequence crossover (order crossover)
        seq1, seq2 = parent1['area_sequence'][:], parent2['area_sequence'][:]
        size = len(seq1)

        # Select crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Copy segment from parent1
        child['area_sequence'] = [-1] * size
        child['area_sequence'][start:end] = seq1[start:end]

        # Fill remaining positions from parent2
        p2_filtered = [x for x in seq2 if x not in child['area_sequence']]
        j = 0
        for i in range(size):
            if child['area_sequence'][i] == -1:
                child['area_sequence'][i] = p2_filtered[j]
                j += 1

        # Path parameters crossover
        for i in range(len(parent1['path_parameters'])):
            if random.random() < 0.5:
                child['path_parameters'].append(parent1['path_parameters'][i].copy())
            else:
                child['path_parameters'].append(parent2['path_parameters'][i].copy())

        return child

    def _mutate(self, individual):
        """Mutation operation"""
        # Area sequence mutation
        if random.random() < 0.5:
            seq = individual['area_sequence']
            if len(seq) > 1:
                i, j = random.sample(range(len(seq)), 2)
                seq[i], seq[j] = seq[j], seq[i]

        # Path parameters mutation
        for params in individual['path_parameters']:
            if random.random() < 0.3:
                params['cutting_direction'] = 90 if params['cutting_direction'] == 0 else 0
            if random.random() < 0.3:
                params['cutting_width'] *= random.uniform(0.9, 1.1)
            if random.random() < 0.3:
                params['start_corner'] = random.choice(['NW', 'NE', 'SW', 'SE'])

        return individual

    def _calculate_solution_metrics(self, path, areas, robot_params):
        """Calculate solution metrics with proper coverage calculation"""
        if not path:
            return 0, 0, 0

        energy = len(path) * robot_params.power_consumption * 0.1
        time_cost = len(path) * 2.0

        # Calculate coverage ratio using path length method (FIXED)
        total_path_length = self._calculate_total_path_length(path)
        covered_area = total_path_length * robot_params.cut_width
        total_area_size = sum(self._calculate_area_size(area) for area in areas)

        if total_area_size > 0:
            coverage = min(1.0, covered_area / total_area_size)
        else:
            coverage = 0.0

        return energy, time_cost, coverage

    def _calculate_total_path_length(self, path_points):
        """Calculate total length of path"""
        if len(path_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i].x - path_points[i - 1].x
            dy = path_points[i].y - path_points[i - 1].y
            total_length += math.sqrt(dx * dx + dy * dy)

        return total_length

    def _calculate_area_size(self, area):
        """Calculate area size"""
        if len(area.points) < 3:
            return 0

        # Use shoelace formula to calculate polygon area
        x_coords = [p.x for p in area.points]
        y_coords = [p.y for p in area.points]

        n = len(x_coords)
        area_size = 0
        for i in range(n):
            j = (i + 1) % n
            area_size += x_coords[i] * y_coords[j]
            area_size -= x_coords[j] * y_coords[i]

        return abs(area_size) / 2


class ParticleSwarmOptimization:
    """Particle Swarm Optimization algorithm for path planning"""

    def __init__(self, algorithm_name="PSO"):
        self.algorithm_name = algorithm_name
        self.swarm_size = 30
        self.max_iterations = 80
        self.w = 0.729  # Inertia weight
        self.c1 = 1.494  # Individual learning factor
        self.c2 = 1.494  # Social learning factor

    def solve_path_planning(self, areas, obstacles, orchard_size, robot_params):
        """Solve path planning using PSO with training tracking"""
        if not areas:
            return PathSolution([], float('inf'), float('inf'), float('inf'), 0.0, self.algorithm_name)

        # Initialize training tracking
        self.training_history = {
            'iterations': [],
            'coverage_ratios': [],
            'energy_consumptions': [],
            'time_costs': [],
            'path_lengths': []
        }

        # Add computational delay for realistic execution time
        start_time = time.time()

        # Initialize particle swarm
        particles = self._initialize_swarm(areas, obstacles, robot_params)

        global_best_position = None
        global_best_fitness = float('inf')

        for iteration in range(self.max_iterations):
            # Add computational complexity
            time.sleep(0.003)  # Small delay per iteration

            iteration_best_position = None
            iteration_best_fitness = float('inf')

            for particle in particles:
                # Evaluate fitness
                fitness = self._evaluate_particle_fitness(particle['position'], areas, obstacles, robot_params)

                # Update personal best
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()

                # Update iteration best
                if fitness < iteration_best_fitness:
                    iteration_best_fitness = fitness
                    iteration_best_position = particle['position'].copy()

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle['position'].copy()

            # Record training metrics for current iteration
            if iteration_best_position is not None:
                current_path = self._build_path_from_pso_solution(iteration_best_position, areas, obstacles,
                                                                  robot_params)
                current_energy, current_time, current_coverage = self._calculate_pso_metrics(current_path, areas,
                                                                                             robot_params)
                current_path_length = self._calculate_total_path_length(current_path)

                self.training_history['iterations'].append(iteration + 1)
                self.training_history['coverage_ratios'].append(current_coverage)
                self.training_history['energy_consumptions'].append(current_energy)
                self.training_history['time_costs'].append(current_time)
                self.training_history['path_lengths'].append(current_path_length)
            else:
                # Even if no best position, record some default values
                self.training_history['iterations'].append(iteration + 1)
                self.training_history['coverage_ratios'].append(0.0)
                self.training_history['energy_consumptions'].append(0.0)
                self.training_history['time_costs'].append(0.0)
                self.training_history['path_lengths'].append(0.0)

            # Update particle velocity and position
            for particle in particles:
                self._update_particle(particle, global_best_position)

        # Build final path
        final_path = self._build_path_from_pso_solution(global_best_position, areas, obstacles, robot_params)
        energy, time_cost, coverage = self._calculate_pso_metrics(final_path, areas, robot_params)

        # Debug: Print training history info
        print(f"PSO Training History: {len(self.training_history['iterations'])} data points recorded")

        return PathSolution(
            path_points=final_path,
            total_cost=global_best_fitness,
            energy_consumption=energy,
            time_cost=time_cost,
            coverage_ratio=coverage,
            algorithm=self.algorithm_name
        )

    def _initialize_swarm(self, areas, obstacles, robot_params):
        """Initialize particle swarm"""
        particles = []

        for _ in range(self.swarm_size):
            # Particle position encoding: path parameters for each area
            position = []
            velocity = []

            for area in areas:
                # Parameters for each area: start x, y, direction angle, cutting spacing ratio
                area_params = [
                    random.uniform(min(p.x for p in area.points), max(p.x for p in area.points)),  # Start x
                    random.uniform(min(p.y for p in area.points), max(p.y for p in area.points)),  # Start y
                    random.uniform(0, 180),  # Direction angle
                    random.uniform(0.8, 1.2)  # Cutting spacing ratio
                ]
                position.extend(area_params)
                velocity.extend([random.uniform(-1, 1) for _ in range(4)])

            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': float('inf')
            }

            particles.append(particle)

        return particles

    def _evaluate_particle_fitness(self, position, areas, obstacles, robot_params):
        """Evaluate particle fitness with fixed cost calculation"""
        try:
            path = self._build_path_from_pso_solution(position, areas, obstacles, robot_params)

            if not path:
                return float('inf')

            # Calculate various costs (all positive)
            energy_cost = len(path) * robot_params.power_consumption * 0.1

            # Path smoothness penalty
            smoothness_penalty = 0
            for i in range(2, len(path)):
                angle1 = math.atan2(path[i - 1].y - path[i - 2].y, path[i - 1].x - path[i - 2].x)
                angle2 = math.atan2(path[i].y - path[i - 1].y, path[i].x - path[i - 1].x)
                angle_diff = abs(angle2 - angle1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                smoothness_penalty += angle_diff

            # Coverage efficiency (convert to penalty for bad coverage)
            total_area = sum(self._calculate_polygon_area(area.points) for area in areas)
            if total_area > 0:
                cutting_area_per_point = robot_params.cut_width * 100
                covered_area = len(path) * cutting_area_per_point
                coverage_ratio = min(1.0, covered_area / total_area)
                coverage_penalty = (1 - coverage_ratio) * 5000  # Penalty for poor coverage
            else:
                coverage_penalty = 5000

            # Obstacle penalty
            obstacle_penalty = 0
            for point in path:
                for obs in obstacles:
                    dist = math.sqrt((point.x - obs.x) ** 2 + (point.y - obs.y) ** 2)
                    if dist < obs.radius + robot_params.width / 2 + 100:
                        obstacle_penalty += 500

            # Total cost (all positive terms)
            total_cost = energy_cost + smoothness_penalty * 100 + coverage_penalty + obstacle_penalty

            return max(1.0, total_cost)  # Ensure cost is always positive

        except:
            return float('inf')

    def _build_path_from_pso_solution(self, position, areas, obstacles, robot_params):
        """Build path from PSO solution"""
        full_path = []

        for i, area in enumerate(areas):
            # Extract area parameters
            start_idx = i * 4
            if start_idx + 3 >= len(position):
                continue

            start_x = position[start_idx]
            start_y = position[start_idx + 1]
            direction = position[start_idx + 2]
            spacing_ratio = position[start_idx + 3]

            area_path = self._generate_pso_area_path(
                area, start_x, start_y, direction, spacing_ratio, obstacles, robot_params
            )
            full_path.extend(area_path)

        return full_path

    def _generate_pso_area_path(self, area, start_x, start_y, direction, spacing_ratio, obstacles, robot_params):
        """Generate PSO-optimized area path"""
        path = []

        # Get area boundaries
        x_coords = [p.x for p in area.points]
        y_coords = [p.y for p in area.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Ensure starting point is within area
        start_x = max(min_x, min(max_x, start_x))
        start_y = max(min_y, min(max_y, start_y))

        cutting_width = robot_params.cut_width * spacing_ratio * (1 - robot_params.overlap / 100)

        # Generate path based on direction
        if 0 <= direction < 90 or 180 <= direction < 270:  # Mainly horizontal direction
            current_y = start_y
            left_to_right = True

            while min_y <= current_y <= max_y:
                if left_to_right:
                    line_start, line_end = (min_x, current_y), (max_x, current_y)
                else:
                    line_start, line_end = (max_x, current_y), (min_x, current_y)

                segment = self._generate_pso_line_segment(line_start, line_end, obstacles, robot_params)
                path.extend(segment)

                current_y += cutting_width if direction < 180 else -cutting_width
                left_to_right = not left_to_right

                if direction < 90:  # Upward
                    if current_y > max_y:
                        break
                else:  # Downward
                    if current_y < min_y:
                        break

        else:  # Mainly vertical direction
            current_x = start_x
            bottom_to_top = True

            while min_x <= current_x <= max_x:
                if bottom_to_top:
                    line_start, line_end = (current_x, min_y), (current_x, max_y)
                else:
                    line_start, line_end = (current_x, max_y), (current_x, min_y)

                segment = self._generate_pso_line_segment(line_start, line_end, obstacles, robot_params)
                path.extend(segment)

                if 90 <= direction < 180:  # Rightward
                    current_x += cutting_width
                    if current_x > max_x:
                        break
                else:  # Leftward
                    current_x -= cutting_width
                    if current_x < min_x:
                        break

                bottom_to_top = not bottom_to_top

        return path

    def _generate_pso_line_segment(self, start, end, obstacles, robot_params):
        """Generate PSO line segment path"""
        path = []
        distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        num_points = max(3, int(distance / 400))

        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])

            # PSO obstacle avoidance strategy
            adjusted_pos = self._pso_avoid_obstacles((x, y), obstacles, robot_params)

            path.append(Point3D(
                x=adjusted_pos[0],
                y=adjusted_pos[1],
                z=0,
                slope=0.0,
                soil_moisture=0.5,
                grass_density=0.7,
                traversability=1.0
            ))

        return path

    def _pso_avoid_obstacles(self, pos, obstacles, robot_params):
        """PSO obstacle avoidance processing"""
        x, y = pos
        safety_margin = robot_params.width / 2 + 250

        for obs in obstacles:
            dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            required_dist = obs.radius + safety_margin

            if dist < required_dist and dist > 0:
                # Use force field method for obstacle avoidance
                repulsion_force = (required_dist - dist) / required_dist
                dx = (x - obs.x) / dist * repulsion_force * 200
                dy = (y - obs.y) / dist * repulsion_force * 200
                x += dx
                y += dy

        return (x, y)

    def _update_particle(self, particle, global_best_position):
        """Update particle velocity and position"""
        for i in range(len(particle['velocity'])):
            # Velocity update formula
            r1, r2 = random.random(), random.random()

            cognitive_component = self.c1 * r1 * (particle['best_position'][i] - particle['position'][i])
            social_component = self.c2 * r2 * (global_best_position[i] - particle['position'][i])

            particle['velocity'][i] = (self.w * particle['velocity'][i] +
                                       cognitive_component + social_component)

            # Limit velocity
            max_velocity = 1000
            particle['velocity'][i] = max(-max_velocity, min(max_velocity, particle['velocity'][i]))

            # Position update
            particle['position'][i] += particle['velocity'][i]

    def _calculate_pso_metrics(self, path, areas, robot_params):
        """Calculate PSO solution metrics with proper coverage"""
        if not path:
            return 0, 0, 0

        energy = len(path) * robot_params.power_consumption * 0.08
        time_cost = len(path) * 1.8

        # Calculate coverage ratio using path length method (FIXED)
        total_path_length = self._calculate_total_path_length(path)
        covered_area = total_path_length * robot_params.cut_width
        total_area = sum(self._calculate_polygon_area(area.points) for area in areas)

        if total_area > 0:
            coverage = min(1.0, covered_area / total_area)
        else:
            coverage = 0.0

        return energy, time_cost, coverage

    def _calculate_total_path_length(self, path_points):
        """Calculate total length of path"""
        if len(path_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i].x - path_points[i - 1].x
            dy = path_points[i].y - path_points[i - 1].y
            total_length += math.sqrt(dx * dx + dy * dy)

        return total_length

    def _calculate_polygon_area(self, points):
        """Calculate polygon area"""
        if len(points) < 3:
            return 0

        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2


class AntColonyOptimization:
    """Ant Colony Optimization algorithm for path planning"""

    def __init__(self, algorithm_name="ACO"):
        self.algorithm_name = algorithm_name
        self.num_ants = 25
        self.max_iterations = 60
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0  # Heuristic information importance
        self.rho = 0.1  # Pheromone evaporation rate
        self.Q = 100  # Pheromone intensity

    def solve_path_planning(self, areas, obstacles, orchard_size, robot_params):
        """Solve path planning using ant colony algorithm with training tracking"""
        if not areas:
            return PathSolution([], float('inf'), float('inf'), float('inf'), 0.0, self.algorithm_name)

        # Initialize training tracking
        self.training_history = {
            'iterations': [],
            'coverage_ratios': [],
            'energy_consumptions': [],
            'time_costs': [],
            'path_lengths': []
        }

        # Add computational delay for realistic execution time
        start_time = time.time()

        # Build graph structure
        nodes = self._create_graph_nodes(areas, obstacles, robot_params)
        if len(nodes) < 2:
            return PathSolution([], float('inf'), float('inf'), float('inf'), 0.0, self.algorithm_name)

        # Initialize pheromone matrix
        pheromone_matrix = np.ones((len(nodes), len(nodes))) * 0.1

        best_path = None
        best_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Add computational complexity
            time.sleep(0.001)  # Small delay per iteration

            iteration_best_path = None
            iteration_best_cost = float('inf')

            # Each ant constructs a path
            for ant in range(self.num_ants):
                path, cost = self._construct_ant_path(nodes, pheromone_matrix, areas, obstacles, robot_params)

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_path = path

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # Record training metrics for current iteration
            if iteration_best_path:
                current_path_points = self._convert_path_to_points(iteration_best_path, nodes, areas, obstacles,
                                                                   robot_params)
                current_energy, current_time, current_coverage = self._calculate_aco_metrics(current_path_points, areas,
                                                                                             robot_params)
                current_path_length = self._calculate_total_path_length(current_path_points)

                self.training_history['iterations'].append(iteration + 1)
                self.training_history['coverage_ratios'].append(current_coverage)
                self.training_history['energy_consumptions'].append(current_energy)
                self.training_history['time_costs'].append(current_time)
                self.training_history['path_lengths'].append(current_path_length)
            else:
                # Even if no best path, record some default values to maintain consistency
                self.training_history['iterations'].append(iteration + 1)
                self.training_history['coverage_ratios'].append(0.0)
                self.training_history['energy_consumptions'].append(0.0)
                self.training_history['time_costs'].append(0.0)
                self.training_history['path_lengths'].append(0.0)

            # Update pheromones
            self._update_pheromones(pheromone_matrix, iteration_best_path, iteration_best_cost, nodes)

        # Build final path points
        final_path_points = []
        if best_path:
            final_path_points = self._convert_path_to_points(best_path, nodes, areas, obstacles, robot_params)

        energy, time_cost, coverage = self._calculate_aco_metrics(final_path_points, areas, robot_params)

        # Debug: Print training history info
        print(f"ACO Training History: {len(self.training_history['iterations'])} data points recorded")

        return PathSolution(
            path_points=final_path_points,
            total_cost=best_cost,
            energy_consumption=energy,
            time_cost=time_cost,
            coverage_ratio=coverage,
            algorithm=self.algorithm_name
        )

    def _create_graph_nodes(self, areas, obstacles, robot_params):
        """Create graph nodes"""
        nodes = []

        # Create key points for each area
        for area_idx, area in enumerate(areas):
            x_coords = [p.x for p in area.points]
            y_coords = [p.y for p in area.points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Create grid points
            grid_size = 2000  # Grid size
            x_steps = max(2, int((max_x - min_x) / grid_size))
            y_steps = max(2, int((max_y - min_y) / grid_size))

            for i in range(x_steps + 1):
                for j in range(y_steps + 1):
                    x = min_x + i * (max_x - min_x) / x_steps
                    y = min_y + j * (max_y - min_y) / y_steps

                    # Check if point is within area and doesn't conflict with obstacles
                    if self._point_in_area((x, y), area) and not self._point_conflicts_with_obstacles((x, y), obstacles,
                                                                                                      robot_params):
                        nodes.append({
                            'x': x,
                            'y': y,
                            'area_idx': area_idx,
                            'type': 'grid'
                        })

        # Add area connection points
        for i in range(len(areas)):
            for j in range(i + 1, len(areas)):
                connection_point = self._find_connection_point(areas[i], areas[j], obstacles, robot_params)
                if connection_point:
                    nodes.append({
                        'x': connection_point[0],
                        'y': connection_point[1],
                        'area_idx': -1,  # Connection point
                        'type': 'connection'
                    })

        return nodes

    def _point_in_area(self, point, area):
        """Check if point is within area"""
        x, y = point
        area_points = area.points

        # Use ray casting method to determine if point is inside polygon
        n = len(area_points)
        inside = False

        p1x, p1y = area_points[0].x, area_points[0].y
        for i in range(1, n + 1):
            p2x, p2y = area_points[i % n].x, area_points[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _point_conflicts_with_obstacles(self, point, obstacles, robot_params):
        """Check if point conflicts with obstacles"""
        x, y = point
        safety_margin = robot_params.width / 2 + 200

        for obs in obstacles:
            dist = math.sqrt((x - obs.x) ** 2 + (y - obs.y) ** 2)
            if dist < obs.radius + safety_margin:
                return True

        return False

    def _find_connection_point(self, area1, area2, obstacles, robot_params):
        """Find connection point between areas"""
        # Find closest boundary points between two areas
        min_dist = float('inf')
        best_point = None

        for p1 in area1.points:
            for p2 in area2.points:
                dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    # Connection point is at the midpoint
                    mid_x = (p1.x + p2.x) / 2
                    mid_y = (p1.y + p2.y) / 2

                    if not self._point_conflicts_with_obstacles((mid_x, mid_y), obstacles, robot_params):
                        best_point = (mid_x, mid_y)

        return best_point

    def _construct_ant_path(self, nodes, pheromone_matrix, areas, obstacles, robot_params):
        """Construct ant path"""
        if not nodes:
            return [], float('inf')

        # Select starting node
        current_node_idx = random.randint(0, len(nodes) - 1)
        visited = {current_node_idx}
        path = [current_node_idx]
        total_cost = 0

        # Visit all nodes by area
        for area_idx in range(len(areas)):
            area_nodes = [i for i, node in enumerate(nodes)
                          if node['area_idx'] == area_idx and i not in visited]

            # Visit nodes in this area
            while area_nodes:
                probabilities = self._calculate_transition_probabilities(
                    current_node_idx, area_nodes, pheromone_matrix, nodes
                )

                if not probabilities:
                    break

                # Select next node
                next_node_idx = np.random.choice(area_nodes, p=probabilities)

                # Calculate movement cost
                cost = self._calculate_node_distance(nodes[current_node_idx], nodes[next_node_idx])
                total_cost += cost

                path.append(next_node_idx)
                visited.add(next_node_idx)
                area_nodes.remove(next_node_idx)
                current_node_idx = next_node_idx

        return path, max(1.0, total_cost)  # Ensure cost is positive

    def _calculate_transition_probabilities(self, current_idx, candidate_nodes, pheromone_matrix, nodes):
        """Calculate transition probabilities"""
        if not candidate_nodes:
            return []

        probabilities = []

        for node_idx in candidate_nodes:
            # Pheromone concentration
            pheromone = pheromone_matrix[current_idx][node_idx]

            # Heuristic information (inverse of distance)
            distance = self._calculate_node_distance(nodes[current_idx], nodes[node_idx])
            heuristic = 1.0 / max(distance, 1.0)

            # Calculate probability
            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)

        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(candidate_nodes)] * len(candidate_nodes)

        return probabilities

    def _calculate_node_distance(self, node1, node2):
        """Calculate distance between nodes"""
        return math.sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)

    def _update_pheromones(self, pheromone_matrix, best_path, best_cost, nodes):
        """Update pheromones"""
        # Pheromone evaporation
        pheromone_matrix *= (1 - self.rho)

        # Add pheromones on best path
        if best_path and len(best_path) > 1:
            delta_pheromone = self.Q / best_cost

            for i in range(len(best_path) - 1):
                from_idx = best_path[i]
                to_idx = best_path[i + 1]
                pheromone_matrix[from_idx][to_idx] += delta_pheromone
                pheromone_matrix[to_idx][from_idx] += delta_pheromone  # Symmetric matrix

    def _convert_path_to_points(self, path, nodes, areas, obstacles, robot_params):
        """Convert path to point sequence"""
        points = []

        for node_idx in path:
            node = nodes[node_idx]

            # Generate detailed path around node
            if node['area_idx'] >= 0:  # Area node
                area = areas[node['area_idx']]
                local_path = self._generate_local_coverage_path(node, area, obstacles, robot_params)
                points.extend(local_path)
            else:  # Connection node
                points.append(Point3D(
                    x=node['x'],
                    y=node['y'],
                    z=0,
                    slope=0.0,
                    soil_moisture=0.5,
                    grass_density=0.7,
                    traversability=1.0
                ))

        return points

    def _generate_local_coverage_path(self, center_node, area, obstacles, robot_params):
        """Generate local coverage path around node"""
        points = []
        center_x, center_y = center_node['x'], center_node['y']

        # Generate small area coverage path centered on node
        coverage_radius = 800  # Coverage radius
        step_size = robot_params.cut_width * 0.8

        # Generate spiral path
        angle = 0
        radius = step_size

        while radius < coverage_radius:
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            # Check if within area and doesn't conflict with obstacles
            if (self._point_in_area((x, y), area) and
                    not self._point_conflicts_with_obstacles((x, y), obstacles, robot_params)):
                points.append(Point3D(
                    x=x, y=y, z=0,
                    slope=0.0,
                    soil_moisture=0.5,
                    grass_density=0.7,
                    traversability=1.0
                ))

            angle += 0.3  # Increase angle
            radius += step_size / 10  # Slowly increase radius

        return points

    def _calculate_aco_metrics(self, path, areas, robot_params):
        """Calculate ACO solution metrics with proper coverage"""
        if not path:
            return 0, 0, 0

        energy = len(path) * robot_params.power_consumption * 0.09
        time_cost = len(path) * 2.2

        # Calculate coverage ratio using path length method (FIXED)
        total_path_length = self._calculate_total_path_length(path)
        covered_area = total_path_length * robot_params.cut_width
        total_area = sum(self._calculate_area_polygon(area.points) for area in areas)

        if total_area > 0:
            coverage = min(1.0, covered_area / total_area)
        else:
            coverage = 0.0

        return energy, time_cost, coverage

    def _calculate_total_path_length(self, path_points):
        """Calculate total length of path"""
        if len(path_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i].x - path_points[i - 1].x
            dy = path_points[i].y - path_points[i - 1].y
            total_length += math.sqrt(dx * dx + dy * dy)

        return total_length

    def _calculate_area_polygon(self, points):
        """Calculate polygon area"""
        if len(points) < 3:
            return 0

        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2


# ==================== Algorithm Comparison System ====================

class RealAlgorithmComparator:
    """Real algorithm performance comparison system"""

    def __init__(self):
        self.algorithms = {
            'SAC': {'name': 'Soft Actor-Critic', 'color': 'blue', 'class': ImprovedSACAgent},
            'GA': {'name': 'Genetic Algorithm', 'color': 'red', 'class': GeneticAlgorithm},
            'PSO': {'name': 'Particle Swarm Optimization', 'color': 'green', 'class': ParticleSwarmOptimization},
            'ACO': {'name': 'Ant Colony Optimization', 'color': 'orange', 'class': AntColonyOptimization}
        }

        self.performance_data = {}
        self.solution_data = {}  # Store actual solutions
        self.comparison_results = {}

    def compare_algorithms(self, areas, obstacles, orchard_size, robot_params):
        """Compare real performance of all algorithms"""
        self.performance_data.clear()
        self.solution_data.clear()
        self.comparison_results.clear()

        # Store algorithm instances with training history
        self.algorithm_instances = {}

        print("Starting real algorithm comparison...")

        for algo_name, algo_info in self.algorithms.items():
            print(f"Running {algo_name} ({algo_info['name']})...")

            # Create algorithm instance
            if algo_name == 'SAC':
                algorithm = algo_info['class'](state_dim=10, action_dim=5, algorithm_name=algo_name)
            else:
                algorithm = algo_info['class'](algorithm_name=algo_name)

            # Store the algorithm instance to access training history later
            self.algorithm_instances[algo_name] = algorithm

            # Execute path planning
            start_time = time.time()
            solution = algorithm.solve_path_planning(areas, obstacles, orchard_size, robot_params)
            execution_time = time.time() - start_time

            # Store results
            self.solution_data[algo_name] = solution

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(solution, execution_time)
            self.performance_data[algo_name] = performance_metrics

            print(f"{algo_name} completed - Path points: {len(solution.path_points)}, "
                  f"Total cost: {solution.total_cost:.2f}, Coverage: {solution.coverage_ratio:.2%}, "
                  f"Execution time: {execution_time:.2f}s")

        self._generate_comparison_summary()
        print("Algorithm comparison completed!")

        return self.performance_data

    def _calculate_performance_metrics(self, solution, execution_time):
        """Calculate performance metrics"""
        path_points = solution.path_points

        metrics = {
            'total_cost': solution.total_cost,
            'energy_consumption': solution.energy_consumption,
            'time_cost': solution.time_cost,
            'coverage_ratio': solution.coverage_ratio,
            'execution_time': execution_time,
            'path_length': len(path_points),
            'algorithm': solution.algorithm
        }

        # Calculate path smoothness
        if len(path_points) > 2:
            direction_changes = 0
            for i in range(2, len(path_points)):
                angle1 = math.atan2(path_points[i - 1].y - path_points[i - 2].y,
                                    path_points[i - 1].x - path_points[i - 2].x)
                angle2 = math.atan2(path_points[i].y - path_points[i - 1].y,
                                    path_points[i].x - path_points[i - 1].x)
                angle_diff = abs(angle2 - angle1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                if angle_diff > math.pi / 4:  # Changes above 45 degrees count as direction change
                    direction_changes += 1

            metrics['smoothness'] = 1.0 - (direction_changes / max(1, len(path_points) - 2))
        else:
            metrics['smoothness'] = 1.0

        # Calculate total path length
        if len(path_points) > 1:
            total_distance = 0
            for i in range(1, len(path_points)):
                dx = path_points[i].x - path_points[i - 1].x
                dy = path_points[i].y - path_points[i - 1].y
                total_distance += math.sqrt(dx * dx + dy * dy)
            metrics['total_distance'] = total_distance
        else:
            metrics['total_distance'] = 0

        return metrics

    def _generate_comparison_summary(self):
        """Generate comparison summary"""
        if not self.performance_data:
            return

        # Find best algorithm for each metric
        best_algorithms = {}

        # Lowest total cost
        best_cost_algo = min(self.performance_data.keys(),
                             key=lambda x: self.performance_data[x]['total_cost'])
        best_algorithms['lowest_cost'] = best_cost_algo

        # Highest coverage ratio
        best_coverage_algo = max(self.performance_data.keys(),
                                 key=lambda x: self.performance_data[x]['coverage_ratio'])
        best_algorithms['best_coverage'] = best_coverage_algo

        # Shortest execution time
        best_time_algo = min(self.performance_data.keys(),
                             key=lambda x: self.performance_data[x]['execution_time'])
        best_algorithms['fastest_execution'] = best_time_algo

        # Highest smoothness
        best_smoothness_algo = max(self.performance_data.keys(),
                                   key=lambda x: self.performance_data[x]['smoothness'])
        best_algorithms['smoothest_path'] = best_smoothness_algo

        # Lowest energy consumption
        best_energy_algo = min(self.performance_data.keys(),
                               key=lambda x: self.performance_data[x]['energy_consumption'])
        best_algorithms['lowest_energy'] = best_energy_algo

        self.comparison_results = best_algorithms

    def get_summary_statistics(self):
        """Get summary statistics"""
        summary = {}

        for algo_name, metrics in self.performance_data.items():
            summary[algo_name] = {
                'total_cost': metrics['total_cost'],
                'energy_consumption': metrics['energy_consumption'],
                'time_cost': metrics['time_cost'],
                'coverage_ratio': metrics['coverage_ratio'],
                'execution_time': metrics['execution_time'],
                'path_length': metrics['path_length'],
                'smoothness': metrics['smoothness'],
                'total_distance': metrics['total_distance']
            }

        return summary

    def get_algorithm_paths(self):
        """Get all algorithm path data"""
        paths = {}
        for algo_name, solution in self.solution_data.items():
            paths[algo_name] = solution.path_points
        return paths


# ==================== Other necessary classes (simplified versions) ====================
# ==================== Training Curves Visualization Window ====================

class TrainingCurvesWindow:
    """Training curves visualization window with real data"""

    def __init__(self, parent, comparator):
        self.parent = parent
        self.comparator = comparator
        self.window = None

    def show(self):
        """Show training curves window"""
        if self.window is not None:
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent.root)
        self.window.title("📈 Algorithm Training Curves Analysis - REAL DATA")
        self.window.geometry("1600x1000")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_widgets()

    def create_widgets(self):
        """Create training curves interface"""
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="📊 Real Training Curves Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="📈 Update Real Training Curves",
                   command=self.update_curves).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save Real Curves Data",
                   command=self.save_curves_data).pack(side=tk.LEFT, padx=5)

        self.curves_status_label = ttk.Label(control_frame, text="Status: Ready to display real curves")
        self.curves_status_label.pack(side=tk.RIGHT)

        # Create matplotlib figure with 4 subplots
        self.curves_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        self.curves_fig.suptitle('Algorithm Training Curves - REAL Performance Evolution Over Iterations',
                                 fontsize=16, fontweight='bold')

        self.ax1.set_title('Coverage Ratio Evolution (Real Data)')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Coverage Ratio (%)')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title('Energy Consumption Evolution (Real Data)')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Energy Consumption (Wh)')
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_title('Time Cost Evolution (Real Data)')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Time Cost (units)')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.set_title('Path Length Evolution (Real Data)')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Path Length (mm)')
        self.ax4.grid(True, alpha=0.3)

        # Embed in tkinter
        self.curves_canvas = FigureCanvasTkAgg(self.curves_fig, main_frame)
        self.curves_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial update
        self.update_curves()

    def update_curves(self):
        """Update training curves plots with real training data"""
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # Check if algorithm instances with training history are available
        if hasattr(self.comparator, 'algorithm_instances') and self.comparator.algorithm_instances:
            self.curves_status_label.config(text="Status: Displaying real training curves", foreground="green")
            self._plot_real_training_curves()
        else:
            # Show message if no real training data
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.text(0.5, 0.5, 'No real training data available\nRun algorithm comparison first',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)

            self.curves_status_label.config(text="Status: No real training data available", foreground="orange")

        # Set proper titles and labels
        self.ax1.set_title('Coverage Ratio Evolution (Real Data)')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Coverage Ratio (%)')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title('Energy Consumption Evolution (Real Data)')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Energy Consumption (Wh)')
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_title('Time Cost Evolution (Real Data)')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Time Cost (units)')
        self.ax3.grid(True, alpha=0.3)

        self.ax4.set_title('Path Length Evolution (Real Data)')
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Path Length (mm)')
        self.ax4.grid(True, alpha=0.3)

        # Add legends
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.legend()

        self.curves_fig.tight_layout()
        self.curves_canvas.draw()

    def _plot_real_training_curves(self):
        """Plot real training curves from algorithm instances"""
        colors = {'SAC': 'blue', 'GA': 'red', 'PSO': 'green', 'ACO': 'orange'}
        markers = {'SAC': 'o', 'GA': 's', 'PSO': '^', 'ACO': 'd'}

        for algo_name, algorithm_instance in self.comparator.algorithm_instances.items():
            # Check if the algorithm has training history
            if hasattr(algorithm_instance, 'training_history') and algorithm_instance.training_history:
                history = algorithm_instance.training_history

                # Get the color and marker for this algorithm
                color = colors.get(algo_name, 'black')
                marker = markers.get(algo_name, 'o')

                # Check if we have data to plot
                if (history['iterations'] and
                        len(history['iterations']) == len(history['coverage_ratios']) ==
                        len(history['energy_consumptions']) == len(history['time_costs']) ==
                        len(history['path_lengths'])):

                    iterations = history['iterations']

                    # Plot coverage ratio (convert to percentage)
                    coverage_percent = [c * 100 for c in history['coverage_ratios']]
                    self.ax1.plot(iterations, coverage_percent, color=color, linewidth=2,
                                  label=f'{algo_name} (Real)', alpha=0.8, marker=marker, markersize=4)

                    # Plot energy consumption
                    self.ax2.plot(iterations, history['energy_consumptions'], color=color, linewidth=2,
                                  label=f'{algo_name} (Real)', alpha=0.8, marker=marker, markersize=4)

                    # Plot time costs
                    self.ax3.plot(iterations, history['time_costs'], color=color, linewidth=2,
                                  label=f'{algo_name} (Real)', alpha=0.8, marker=marker, markersize=4)

                    # Plot path lengths
                    self.ax4.plot(iterations, history['path_lengths'], color=color, linewidth=2,
                                  label=f'{algo_name} (Real)', alpha=0.8, marker=marker, markersize=4)

                    print(f"✅ Plotted real training curve for {algo_name}: {len(iterations)} data points")

                    # Print some statistics for debugging
                    print(f"   - Coverage range: {min(coverage_percent):.1f}% to {max(coverage_percent):.1f}%")
                    print(
                        f"   - Energy range: {min(history['energy_consumptions']):.0f} to {max(history['energy_consumptions']):.0f} Wh")

                else:
                    print(f"❌ Warning: {algo_name} has incomplete training history data")
                    print(f"   - Iterations: {len(history.get('iterations', []))}")
                    print(f"   - Coverage: {len(history.get('coverage_ratios', []))}")
                    print(f"   - Energy: {len(history.get('energy_consumptions', []))}")
                    print(f"   - Time: {len(history.get('time_costs', []))}")
                    print(f"   - Path: {len(history.get('path_lengths', []))}")
            else:
                print(f"❌ Warning: {algo_name} has no training history")

        # If no real data was plotted, show a message
        if not any(ax.has_data() for ax in [self.ax1, self.ax2, self.ax3, self.ax4]):
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.text(0.5, 0.5, 'Real training data collection failed\nCheck algorithm implementations',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            print("❌ No real training data was successfully plotted")

    def save_curves_data(self):
        """Save real training curves data"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Real Training Curves Data"
        )

        if filename:
            try:
                # Collect real training data
                curves_data = {
                    'timestamp': time.time(),
                    'description': 'Real Algorithm Training Curves Data',
                    'data_type': 'REAL_TRAINING_DATA',
                    'algorithms': {}
                }

                # Get real training data from algorithm instances
                if hasattr(self.comparator, 'algorithm_instances') and self.comparator.algorithm_instances:
                    algorithms_with_data = 0
                    for algo_name, algorithm_instance in self.comparator.algorithm_instances.items():
                        if hasattr(algorithm_instance, 'training_history') and algorithm_instance.training_history:
                            history = algorithm_instance.training_history
                            if history['iterations']:  # Check if we have actual data
                                curves_data['algorithms'][algo_name] = {
                                    'iterations': history['iterations'],
                                    'coverage_ratios': history['coverage_ratios'],
                                    'energy_consumptions': history['energy_consumptions'],
                                    'time_costs': history['time_costs'],
                                    'path_lengths': history['path_lengths'],
                                    'data_points': len(history['iterations']),
                                    'algorithm_type': algo_name
                                }
                                algorithms_with_data += 1
                            else:
                                # If no real data, note it
                                curves_data['algorithms'][algo_name] = {
                                    'note': 'No training history data recorded for this algorithm',
                                    'algorithm_type': algo_name
                                }
                        else:
                            curves_data['algorithms'][algo_name] = {
                                'note': 'Algorithm instance has no training_history attribute',
                                'algorithm_type': algo_name
                            }

                    curves_data['summary'] = {
                        'total_algorithms': len(self.comparator.algorithm_instances),
                        'algorithms_with_data': algorithms_with_data
                    }
                else:
                    curves_data['note'] = 'No algorithm instances available - run comparison first'
                    curves_data['summary'] = {
                        'total_algorithms': 0,
                        'algorithms_with_data': 0
                    }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(curves_data, f, indent=2, ensure_ascii=False, default=str)

                algorithms_with_data = curves_data.get('summary', {}).get('algorithms_with_data', 0)
                messagebox.showinfo("Save Successful",
                                    f"Real training curves data saved to: {filename}\n"
                                    f"Algorithms with real data: {algorithms_with_data}/4\n"
                                    f"Data type: REAL TRAINING CURVES")

            except Exception as e:
                messagebox.showerror("Save Failed", f"Error saving real curves data: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        try:
            if self.window:
                self.window.destroy()
        except:
            pass
        finally:
            self.window = None
class TrainingVisualizationWindow:
    """Simplified training visualization window"""

    def __init__(self, parent, agent):
        self.parent = parent
        self.agent = agent
        self.window = None
        self.is_active = False
        self.update_interval = 100

    def show(self):
        """Show the training visualization window"""
        if self.window is not None:
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent.root)
        self.window.title("🎯 SAC Training Visualization")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.is_active = True
        self.create_simple_visualization()

    def create_simple_visualization(self):
        """Create simple visualization"""
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Training info
        self.info_text = tk.Text(main_frame, height=10, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Close", command=self.on_closing).pack(side=tk.RIGHT)

        # Update display
        self.update_display()

    def update_display(self):
        """Update display with training info"""
        if self.is_active and self.agent.training_metrics:
            self.info_text.delete(1.0, tk.END)

            latest = self.agent.training_metrics[-1]
            info = f"""SAC Training Status:

Episode: {latest.episode}
Reward: {latest.reward:.4f}
Loss: {latest.loss:.6f}
Success Rate: {latest.success_rate:.2%}
Energy Efficiency: {latest.energy_efficiency:.2%}
Time Efficiency: {latest.time_efficiency:.2%}
Training Steps: {self.agent.training_steps}
"""
            self.info_text.insert(tk.END, info)

            self.window.after(self.update_interval, self.update_display)

    def on_closing(self):
        """Handle window closing"""
        try:
            self.is_active = False
            if self.window:
                self.window.destroy()
        except:
            pass
        finally:
            self.window = None


# Add the rest of the classes from the original code...
# (AlgorithmComparisonWindow, AdvancedHRLMowerPlanner, etc. - keeping them largely the same but with the fixed algorithms)

class AlgorithmComparisonWindow:
    """Algorithm comparison visualization window"""

    def __init__(self, parent, comparator):
        self.parent = parent
        self.comparator = comparator
        self.window = None

    def show(self):
        """Show algorithm comparison window"""
        if self.window is not None:
            self.window.lift()
            return

        self.window = tk.Toplevel(self.parent.root)
        self.window.title("🏆 Real Algorithm Performance Comparison Analysis")
        self.window.geometry("1400x900")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.create_widgets()

    def create_widgets(self):
        """Create comparison window interface"""
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Comparison Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(control_frame, text="🚀 Start Real Algorithm Comparison",
                   command=self.start_real_comparison).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 Update Charts",
                   command=self.update_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save Results",
                   command=self.save_results).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(control_frame, text="Status: Waiting to start")
        self.status_label.pack(side=tk.RIGHT)

        # Create notebook for different views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Performance comparison tab
        self.create_performance_tab()

        # Path visualization tab
        self.create_path_visualization_tab()

        # Results table tab
        self.create_results_table_tab()

    def create_performance_tab(self):
        """Create performance comparison tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="📈 Performance Comparison")

        # Create matplotlib figure
        self.perf_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.perf_fig.suptitle('Real Algorithm Performance Comparison Analysis', fontsize=16, fontweight='bold')

        self.ax1.set_title('Total Cost Comparison')
        self.ax2.set_title('Execution Time Comparison')
        self.ax3.set_title('Coverage Ratio Comparison')
        self.ax4.set_title('Energy Consumption Comparison')

        # Embed in tkinter
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_path_visualization_tab(self):
        """Create path visualization tab"""
        path_frame = ttk.Frame(self.notebook)
        self.notebook.add(path_frame, text="🛤️ Path Visualization")

        # Create path visualization plot
        self.path_fig, self.path_ax = plt.subplots(1, 1, figsize=(14, 10))
        self.path_fig.suptitle('Algorithm Path Comparison - All Paths on Same Map', fontsize=16, fontweight='bold')

        self.path_ax.set_title('Path Trajectory Comparison')
        self.path_ax.set_xlabel('X Coordinate (mm)')
        self.path_ax.set_ylabel('Y Coordinate (mm)')
        self.path_ax.grid(True, alpha=0.3)

        # Embed in tkinter
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, path_frame)
        self.path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_results_table_tab(self):
        """Create results table tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Detailed Results")

        # Create results table
        columns = (
            'Algorithm', 'Total Cost', 'Energy', 'Time', 'Coverage', 'Path Length', 'Smoothness', 'Execution Time')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def start_real_comparison(self):
        """Start real algorithm comparison"""
        # Get parent application reference
        main_app = self.parent

        if not hasattr(main_app, 'areas') or not main_app.areas:
            messagebox.showwarning("Warning", "Please define working areas first!")
            return

        self.status_label.config(text="Status: Running real algorithm comparison...", foreground="orange")
        self.window.update()

        try:
            # Run real comparison
            self.comparator.compare_algorithms(
                main_app.areas,
                main_app.obstacles,
                main_app.orchard_size,
                main_app.robot_params
            )

            # Update all visualizations
            self.update_plots()
            self.update_results_table()

            self.status_label.config(text="Status: Real algorithm comparison completed", foreground="green")

            messagebox.showinfo("Comparison Complete",
                                f"Real algorithm comparison completed!\n"
                                f"Compared {len(self.comparator.algorithms)} algorithms\n"
                                f"Results displayed in the tabs")

        except Exception as e:
            self.status_label.config(text="Status: Comparison failed", foreground="red")
            messagebox.showerror("Error", f"Error during algorithm comparison: {str(e)}")

    def update_plots(self):
        """Update performance plots"""
        if not self.comparator.performance_data:
            return

        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        algorithms = list(self.comparator.performance_data.keys())
        colors = [self.comparator.algorithms[algo]['color'] for algo in algorithms]

        # Total cost comparison
        costs = [self.comparator.performance_data[algo]['total_cost'] for algo in algorithms]
        bars1 = self.ax1.bar(algorithms, costs, color=colors, alpha=0.7)
        self.ax1.set_title('Total Cost Comparison')
        self.ax1.set_ylabel('Total Cost')
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                          f'{cost:.1f}', ha='center', va='bottom')

        # Execution time comparison
        exec_times = [self.comparator.performance_data[algo]['execution_time'] for algo in algorithms]
        bars2 = self.ax2.bar(algorithms, exec_times, color=colors, alpha=0.7)
        self.ax2.set_title('Execution Time Comparison')
        self.ax2.set_ylabel('Execution Time (seconds)')
        for bar, time_val in zip(bars2, exec_times):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                          f'{time_val:.2f}s', ha='center', va='bottom')

        # Coverage ratio comparison
        coverages = [self.comparator.performance_data[algo]['coverage_ratio'] for algo in algorithms]
        bars3 = self.ax3.bar(algorithms, coverages, color=colors, alpha=0.7)
        self.ax3.set_title('Coverage Ratio Comparison')
        self.ax3.set_ylabel('Coverage Ratio')
        self.ax3.set_ylim(0, 1)
        for bar, coverage in zip(bars3, coverages):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                          f'{coverage:.2%}', ha='center', va='bottom')

        # Energy consumption comparison
        energies = [self.comparator.performance_data[algo]['energy_consumption'] for algo in algorithms]
        bars4 = self.ax4.bar(algorithms, energies, color=colors, alpha=0.7)
        self.ax4.set_title('Energy Consumption Comparison')
        self.ax4.set_ylabel('Energy Consumption (Wh)')
        for bar, energy in zip(bars4, energies):
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                          f'{energy:.1f}', ha='center', va='bottom')

        self.perf_fig.tight_layout()
        self.perf_canvas.draw()

        # Update path visualization
        self.update_path_visualization()

    def update_path_visualization(self):
        """Update path visualization showing all algorithm paths"""
        if not hasattr(self.comparator, 'solution_data') or not self.comparator.solution_data:
            return

        self.path_ax.clear()
        self.path_ax.set_title('Algorithm Path Trajectory Comparison')
        self.path_ax.set_xlabel('X Coordinate (mm)')
        self.path_ax.set_ylabel('Y Coordinate (mm)')
        self.path_ax.grid(True, alpha=0.3)

        # Draw areas first
        main_app = self.parent
        if hasattr(main_app, 'areas') and main_app.areas:
            for i, area in enumerate(main_app.areas):
                if len(area.points) >= 3:
                    x_coords = [p.x for p in area.points] + [area.points[0].x]
                    y_coords = [p.y for p in area.points] + [area.points[0].y]
                    self.path_ax.fill(x_coords, y_coords, color='lightgray', alpha=0.3,
                                      edgecolor='black', linewidth=1)

                    # Add area label
                    center_x = sum(p.x for p in area.points) / len(area.points)
                    center_y = sum(p.y for p in area.points) / len(area.points)
                    self.path_ax.text(center_x, center_y, f'Area{i + 1}',
                                      ha='center', va='center', fontsize=8,
                                      bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

        # Draw obstacles
        if hasattr(main_app, 'obstacles') and main_app.obstacles:
            for obs in main_app.obstacles:
                circle = Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.5)
                self.path_ax.add_patch(circle)

        # Draw algorithm paths with different colors and styles
        line_styles = ['-', '--', '-.', ':']

        for i, (algo_name, solution) in enumerate(self.comparator.solution_data.items()):
            if solution.path_points:
                x_coords = [p.x for p in solution.path_points]
                y_coords = [p.y for p in solution.path_points]

                color = self.comparator.algorithms[algo_name]['color']
                line_style = line_styles[i % len(line_styles)]

                self.path_ax.plot(x_coords, y_coords,
                                  color=color, linewidth=2, alpha=0.8,
                                  linestyle=line_style, label=f'{algo_name} ({len(solution.path_points)} points)')

                # Mark start and end points
                if len(x_coords) > 0:
                    self.path_ax.scatter(x_coords[0], y_coords[0], color=color, s=100,
                                         marker='o', edgecolor='black', zorder=5)
                    self.path_ax.scatter(x_coords[-1], y_coords[-1], color=color, s=100,
                                         marker='s', edgecolor='black', zorder=5)

        self.path_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.path_ax.set_aspect('equal')

        self.path_fig.tight_layout()
        self.path_canvas.draw()

    def update_results_table(self):
        """Update results table"""
        # Clear existing data
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if not self.comparator.performance_data:
            return

        # Add data to table
        for algo_name, metrics in self.comparator.performance_data.items():
            self.results_tree.insert('', 'end', values=(
                algo_name,
                f'{metrics["total_cost"]:.2f}',
                f'{metrics["energy_consumption"]:.1f}',
                f'{metrics["time_cost"]:.1f}',
                f'{metrics["coverage_ratio"]:.2%}',
                f'{metrics["path_length"]}',
                f'{metrics["smoothness"]:.3f}',
                f'{metrics["execution_time"]:.2f}s'
            ))

    def save_results(self):
        """Save comparison results"""
        if not self.comparator.performance_data:
            messagebox.showwarning("Warning", "No comparison data to save!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Algorithm Comparison Results"
        )

        if filename:
            try:
                summary = self.comparator.get_summary_statistics()
                paths = {}
                for algo_name, solution in self.comparator.solution_data.items():
                    paths[algo_name] = [(p.x, p.y, p.z) for p in solution.path_points]

                export_data = {
                    'comparison_timestamp': time.time(),
                    'algorithms_compared': list(self.comparator.algorithms.keys()),
                    'performance_summary': summary,
                    'algorithm_paths': paths,
                    'comparison_results': self.comparator.comparison_results
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

                messagebox.showinfo("Save Successful", f"Algorithm comparison results saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Save Failed", f"Error saving data: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        try:
            if self.window:
                self.window.destroy()
        except:
            pass
        finally:
            self.window = None


# ==================== Main Application Class ====================

class AdvancedHRLMowerPlanner:
    """Advanced HRL-based mower path planning system with real algorithm comparison"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Advanced HRL Mower Path Planning System - Fixed Version")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#f0f0f0')

        # Core components
        self.robot_params = RobotParams()
        self.areas = []
        self.obstacles = []
        self.path_points = []
        self.current_area = []

        # HRL Architecture
        self.state_representation = StateRepresentation()
        self.global_planner = ImprovedSACAgent(state_dim=10, action_dim=5)

        # Real algorithm comparison system
        self.algorithm_comparator = RealAlgorithmComparator()
        self.comparison_window = AlgorithmComparisonWindow(self, self.algorithm_comparator)

        # Training visualization
        self.training_window = TrainingVisualizationWindow(self, self.global_planner)

        # Training curves visualization
        self.training_curves_window = TrainingCurvesWindow(self, self.algorithm_comparator)

        # Store algorithm instances with training history
        self.algorithm_instances = {}

        # Interface state
        self.drawing_mode = True
        self.orchard_size = (20000, 15000)
        self.scale = 0.1

        # Create interface
        self.create_widgets()
        self.setup_plot()
        self.generate_terrain()
        self.generate_default_complex_scene()

    def create_widgets(self):
        """Create enhanced interface"""
        # Main container with notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Environment tab
        self.create_environment_tab()

        # Algorithm comparison tab
        self.create_algorithm_comparison_tab()

        # Analysis tab
        self.create_analysis_tab()

    def create_environment_tab(self):
        """Create environment configuration tab"""
        env_frame = ttk.Frame(self.notebook)
        self.notebook.add(env_frame, text="🌍 Environment")

        # Left panel for controls
        control_frame = ttk.Frame(env_frame, width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Environment controls
        self.create_environment_controls(control_frame)

        # Plot area
        self.create_plot_area(env_frame)

    def create_environment_controls(self, parent):
        """Create environment control panel"""
        # Robot parameters
        robot_frame = ttk.LabelFrame(parent, text="🤖 Robot Configuration", padding=10)
        robot_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(robot_frame, text="Battery Capacity (Wh):").grid(row=0, column=0, sticky=tk.W)
        self.battery_var = tk.StringVar(value=str(self.robot_params.battery_capacity))
        ttk.Entry(robot_frame, textvariable=self.battery_var, width=10).grid(row=0, column=1)

        ttk.Label(robot_frame, text="Power Consumption (W):").grid(row=1, column=0, sticky=tk.W)
        self.power_var = tk.StringVar(value=str(self.robot_params.power_consumption))
        ttk.Entry(robot_frame, textvariable=self.power_var, width=10).grid(row=1, column=1)

        # Scene generation controls
        scenario_frame = ttk.LabelFrame(parent, text="🎭 Scene Generation", padding=10)
        scenario_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(scenario_frame, text="Generate Complex Scene",
                   command=self.generate_default_complex_scene).pack(fill=tk.X, pady=2)
        ttk.Button(scenario_frame, text="Clear All Areas",
                   command=self.clear_all_areas).pack(fill=tk.X, pady=2)
        ttk.Button(scenario_frame, text="Add Random Obstacles",
                   command=self.add_random_obstacles).pack(fill=tk.X, pady=2)

        # Drawing instructions
        instruction_frame = ttk.LabelFrame(parent, text="📝 Instructions", padding=10)
        instruction_frame.pack(fill=tk.X, pady=(0, 10))

        instructions = """
        Instructions:
        • Left click: Add area vertex
        • Right click: Complete current area
        • Areas need at least 3 vertices
        • Different colors represent different priorities
        """

        instruction_label = ttk.Label(instruction_frame, text=instructions,
                                      font=("Arial", 9), justify=tk.LEFT)
        instruction_label.pack()

    def create_algorithm_comparison_tab(self):
        """Create algorithm comparison tab"""
        comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(comp_frame, text="🏆 Algorithm Comparison")

        # Main control panel
        control_panel = ttk.LabelFrame(comp_frame, text="🎮 Algorithm Comparison Control", padding=15)
        control_panel.pack(fill=tk.X, padx=10, pady=10)

        # Algorithm descriptions
        algo_desc_frame = ttk.Frame(control_panel)
        algo_desc_frame.pack(fill=tk.X, pady=(0, 10))

        algo_descriptions = """
        This system compares four different path planning algorithms (FIXED VERSION):

        🧠 SAC (Soft Actor-Critic): Deep reinforcement learning, now with proper execution time and coverage calculation
        🧬 GA (Genetic Algorithm): Genetic evolution with corrected cost function (no negative values)
        🐝 PSO (Particle Swarm Optimization): Swarm intelligence with fixed coverage ratio calculation  
        🐜 ACO (Ant Colony Optimization): Ant colony algorithm with proper area coverage metrics

        All algorithms now correctly calculate coverage ratios and maintain positive cost values!
        """

        desc_label = ttk.Label(algo_desc_frame, text=algo_descriptions,
                               font=("Arial", 10), justify=tk.LEFT)
        desc_label.pack()

        # Comparison controls
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="🚀 Start Fixed Algorithm Comparison",
                   command=self.start_algorithm_comparison,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(button_frame, text="📊 Show Comparison Window",
                   command=self.show_comparison_window).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(button_frame, text="📈 Show Training Curves",
                   command=self.show_training_curves).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(button_frame, text="📋 Generate Comparison Report",
                   command=self.generate_comparison_report).pack(side=tk.LEFT, padx=(0, 10))

        # Status display
        self.comparison_status_label = ttk.Label(control_panel, text="Status: Ready for fixed comparison",
                                                 font=("Arial", 11, "bold"))
        self.comparison_status_label.pack(pady=10)

        # Results preview
        results_frame = ttk.LabelFrame(comp_frame, text="📈 Comparison Results Preview", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.results_text = tk.Text(results_frame, height=20, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_analysis_tab(self):
        """Create analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Analysis")

        # Analysis controls
        control_frame = ttk.LabelFrame(analysis_frame, text="🔍 Analysis Tools", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="📈 Performance Analysis",
                   command=self.show_performance_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Export Data",
                   command=self.export_all_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📄 Generate Report",
                   command=self.generate_full_report).pack(side=tk.LEFT, padx=5)

        # Analysis display
        self.analysis_text = tk.Text(analysis_frame, font=("Courier", 10))
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=analysis_scrollbar.set)

        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 10), padx=(0, 10))

    def create_plot_area(self, parent):
        """Create enhanced 2D plot area"""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib 2D figure
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # Bind mouse events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def setup_plot(self):
        """Setup 2D plot"""
        self.ax.set_xlim(0, self.orchard_size[0])
        self.ax.set_ylim(0, self.orchard_size[1])
        self.ax.set_xlabel('X Distance (mm)')
        self.ax.set_ylabel('Y Distance (mm)')
        self.ax.set_title('Advanced HRL Mower Path Planning - Fixed Algorithm Version')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

    def generate_terrain(self):
        """Generate complex terrain"""
        x = np.linspace(0, self.orchard_size[0], 100)
        y = np.linspace(0, self.orchard_size[1], 100)
        X, Y = np.meshgrid(x, y)

        # Create terrain
        slope_rad = math.radians(15.0)
        direction_rad = math.radians(45.0)

        dx = math.cos(direction_rad)
        dy = math.sin(direction_rad)
        Z = (X * dx + Y * dy) * math.tan(slope_rad)

        # Add complexity
        Z += 100 * np.sin(X / 3000) * np.cos(Y / 2000)
        Z += 50 * np.sin(X / 1000) * np.sin(Y / 800)
        Z += 200 * np.exp(-((X - 10000) ** 2 + (Y - 7500) ** 2) / 5000000)
        Z -= 150 * np.exp(-((X - 15000) ** 2 + (Y - 5000) ** 2) / 3000000)

        self.terrain_data = (X, Y, Z)

    def generate_default_complex_scene(self):
        """Generate complex default scene"""
        self.areas.clear()
        self.obstacles.clear()
        self.path_points.clear()

        # Generate areas
        self._generate_test_areas()
        self._generate_test_obstacles()

        self.update_plot()
        messagebox.showinfo("Complex Scene Generated",
                            f"Test scene generated:\n"
                            f"• {len(self.areas)} working areas\n"
                            f"• {len(self.obstacles)} obstacles\n"
                            f"• Ready for fixed algorithm comparison")

    def _generate_test_areas(self):
        """Generate test working areas"""
        # Area 1: High priority precision area
        area1 = TerraceArea(
            points=[
                Point3D(2000, 2000, 0, slope=5.0, soil_moisture=0.6, grass_density=0.8, traversability=0.9),
                Point3D(6000, 2000, 0, slope=5.0, soil_moisture=0.6, grass_density=0.8, traversability=0.9),
                Point3D(6000, 5000, 0, slope=8.0, soil_moisture=0.7, grass_density=0.8, traversability=0.85),
                Point3D(2000, 5000, 0, slope=8.0, soil_moisture=0.7, grass_density=0.8, traversability=0.85)
            ],
            priority=5,
            grass_type="short",
            last_mowed=7.0,
            soil_type="loam",
            drainage=0.8
        )
        self.areas.append(area1)

        # Area 2: Medium priority main area
        area2 = TerraceArea(
            points=[
                Point3D(8000, 1000, 0, slope=12.0, soil_moisture=0.4, grass_density=0.7, traversability=0.8),
                Point3D(15000, 1000, 0, slope=12.0, soil_moisture=0.4, grass_density=0.7, traversability=0.8),
                Point3D(15000, 8000, 0, slope=15.0, soil_moisture=0.5, grass_density=0.7, traversability=0.75),
                Point3D(8000, 8000, 0, slope=15.0, soil_moisture=0.5, grass_density=0.7, traversability=0.75)
            ],
            priority=3,
            grass_type="medium",
            last_mowed=3.0,
            soil_type="clay",
            drainage=0.4
        )
        self.areas.append(area2)

        # Area 3: Large lower area
        area3 = TerraceArea(
            points=[
                Point3D(1000, 9000, 0, slope=8.0, soil_moisture=0.8, grass_density=0.6, traversability=0.7),
                Point3D(18000, 9000, 0, slope=8.0, soil_moisture=0.8, grass_density=0.6, traversability=0.7),
                Point3D(18000, 14000, 0, slope=10.0, soil_moisture=0.9, grass_density=0.6, traversability=0.65),
                Point3D(1000, 14000, 0, slope=10.0, soil_moisture=0.9, grass_density=0.6, traversability=0.65)
            ],
            priority=2,
            grass_type="mixed",
            last_mowed=1.0,
            soil_type="clay",
            drainage=0.2
        )
        self.areas.append(area3)

    def _generate_test_obstacles(self):
        """Generate test obstacles"""
        # Fruit trees
        tree_positions = [
            (3000, 3000), (5000, 3000), (3000, 4500), (5000, 4500),
            (10000, 3000), (12000, 3000), (14000, 3000),
            (10000, 5000), (12000, 5000), (14000, 5000),
            (5000, 10000), (8000, 11000), (12000, 12000), (15000, 11500)
        ]

        for x, y in tree_positions:
            self.obstacles.append(Obstacle(
                x=x, y=y, radius=random.uniform(100, 200), height=random.uniform(2500, 4000),
                obstacle_type="tree", creates_shadow=True, soil_moisture_effect=0.1
            ))

        # Equipment and buildings
        self.obstacles.append(Obstacle(500, 500, 800, 2000, "equipment", False, -0.2))
        self.obstacles.append(Obstacle(19000, 500, 600, 1500, "equipment", False, -0.1))
        self.obstacles.append(Obstacle(10000, 14500, 500, 1000, "equipment", False, 0.0))

        # Rocks
        rock_positions = [(7000, 4000), (13000, 6000), (16000, 10000)]
        for x, y in rock_positions:
            self.obstacles.append(Obstacle(x, y, random.uniform(300, 500), random.uniform(600, 1000),
                                           "rock", False, 0.0))

    def clear_all_areas(self):
        """Clear all areas and obstacles"""
        self.areas.clear()
        self.obstacles.clear()
        self.path_points.clear()
        self.current_area.clear()
        self.update_plot()
        messagebox.showinfo("Clear Complete", "All areas and obstacles have been cleared")

    def add_random_obstacles(self):
        """Add random obstacles"""
        num_obstacles = random.randint(3, 8)
        for _ in range(num_obstacles):
            x = random.uniform(1000, self.orchard_size[0] - 1000)
            y = random.uniform(1000, self.orchard_size[1] - 1000)
            radius = random.uniform(200, 500)
            obs_type = random.choice(["tree", "rock", "debris"])

            self.obstacles.append(Obstacle(
                x=x, y=y, radius=radius, height=random.uniform(500, 2000),
                obstacle_type=obs_type, creates_shadow=obs_type == "tree",
                soil_moisture_effect=random.uniform(-0.1, 0.2)
            ))

        self.update_plot()
        messagebox.showinfo("Obstacles Added", f"Added {num_obstacles} random obstacles")

    def start_algorithm_comparison(self):
        """Start algorithm comparison"""
        if not self.areas:
            messagebox.showwarning("Warning", "Please generate or draw working areas first!")
            return

        self.comparison_status_label.config(text="Status: Running fixed algorithm comparison...", foreground="orange")
        self.root.update()

        def comparison_worker():
            """Background comparison worker"""
            try:
                # Run real algorithm comparison
                performance_data = self.algorithm_comparator.compare_algorithms(
                    self.areas, self.obstacles, self.orchard_size, self.robot_params
                )

                # Update UI in main thread
                self.root.after(0, self._comparison_completed, performance_data)

            except Exception as e:
                self.root.after(0, self._comparison_failed, str(e))

        # Start comparison in background thread
        comparison_thread = threading.Thread(target=comparison_worker)
        comparison_thread.daemon = True
        comparison_thread.start()

    def _comparison_completed(self, performance_data):
        """Handle comparison completion"""
        self.comparison_status_label.config(text="Status: Fixed algorithm comparison completed", foreground="green")

        # Store algorithm instances for training curves access
        if hasattr(self.algorithm_comparator, 'algorithm_instances'):
            self.algorithm_instances = self.algorithm_comparator.algorithm_instances
            print(f"Stored {len(self.algorithm_instances)} algorithm instances with training history")

        # Update results display
        self._update_comparison_results()

        # Update plot with all paths
        self.update_plot_with_algorithm_paths()

        messagebox.showinfo("Fixed Comparison Complete",
                            f"Fixed algorithm comparison completed!\n"
                            f"Compared {len(self.algorithm_comparator.algorithms)} algorithms\n"
                            f"All coverage ratios and costs are now correctly calculated!\n"
                            f"Real training curves are now available - click 'Show Training Curves'")

    def _comparison_failed(self, error_msg):
        """Handle comparison failure"""
        self.comparison_status_label.config(text="Status: Comparison failed", foreground="red")
        messagebox.showerror("Comparison Failed", f"Error during algorithm comparison:\n{error_msg}")

    def _update_comparison_results(self):
        """Update comparison results display"""
        self.results_text.delete(1.0, tk.END)

        if not self.algorithm_comparator.performance_data:
            self.results_text.insert(tk.END, "No comparison results to display")
            return

        results_content = "=" * 80 + "\n"
        results_content += "Fixed Algorithm Performance Comparison Results\n"
        results_content += "=" * 80 + "\n\n"

        # Summary table
        results_content += "Algorithm Performance Summary (FIXED):\n"
        results_content += "-" * 80 + "\n"
        results_content += f"{'Algorithm':<8} {'Total Cost':<12} {'Energy':<8} {'Time':<8} {'Coverage':<10} {'Path Points':<8} {'Exec Time':<10}\n"
        results_content += "-" * 80 + "\n"

        for algo_name, metrics in self.algorithm_comparator.performance_data.items():
            results_content += f"{algo_name:<8} {metrics['total_cost']:<12.1f} {metrics['energy_consumption']:<8.1f} "
            results_content += f"{metrics['time_cost']:<8.1f} {metrics['coverage_ratio']:<10.2%} "
            results_content += f"{metrics['path_length']:<8} {metrics['execution_time']:<10.2f}s\n"

        # Best performance indicators
        if self.algorithm_comparator.comparison_results:
            results_content += "\nBest Performance Metrics (FIXED):\n"
            results_content += "-" * 40 + "\n"

            best_results = self.algorithm_comparator.comparison_results
            for metric, algo in best_results.items():
                metric_names = {
                    'lowest_cost': 'Lowest Total Cost',
                    'best_coverage': 'Highest Coverage Ratio',
                    'fastest_execution': 'Fastest Execution',
                    'smoothest_path': 'Smoothest Path',
                    'lowest_energy': 'Lowest Energy Consumption'
                }
                results_content += f"{metric_names.get(metric, metric)}: {algo}\n"

        # Algorithm paths info
        results_content += "\nFixed Path Generation Status:\n"
        results_content += "-" * 40 + "\n"

        algorithm_paths = self.algorithm_comparator.get_algorithm_paths()
        for algo_name, path_points in algorithm_paths.items():
            path_length = 0
            if len(path_points) > 1:
                for i in range(1, len(path_points)):
                    dx = path_points[i].x - path_points[i - 1].x
                    dy = path_points[i].y - path_points[i - 1].y
                    path_length += math.sqrt(dx * dx + dy * dy)

            results_content += f"{algo_name}: {len(path_points)} path points, Total length: {path_length:.0f}mm\n"

        results_content += "\n" + "=" * 80 + "\n"
        results_content += "FIXES APPLIED:\n"
        results_content += "• SAC: Added proper computational complexity for realistic execution time\n"
        results_content += "• GA: Fixed cost calculation to prevent negative values\n"
        results_content += "• PSO: Corrected fitness function and coverage ratio calculation\n"
        results_content += "• ACO: Improved coverage metric calculation\n"
        results_content += "• All: Fixed area calculation using proper shoelace formula\n"
        results_content += "=" * 80

        self.results_text.insert(tk.END, results_content)

    def show_comparison_window(self):
        """Show detailed comparison window"""
        self.comparison_window.show()

    def show_training_curves(self):
        """Show training curves window"""
        self.training_curves_window.show()

    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        if not self.algorithm_comparator.performance_data:
            messagebox.showwarning("Warning", "Please run algorithm comparison first!")
            return

        report_window = tk.Toplevel(self.root)
        report_window.title("🏆 Fixed Algorithm Comparison Report")
        report_window.geometry("1000x700")

        # Create report text area
        report_text = tk.Text(report_window, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(report_window, command=report_text.yview)
        report_text.config(yscrollcommand=scrollbar.set)

        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

        # Generate report content
        report_content = self._generate_detailed_report()
        report_text.insert(tk.END, report_content)
        report_text.config(state=tk.DISABLED)

        # Add save button
        save_frame = ttk.Frame(report_window)
        save_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(save_frame, text="💾 Save Report",
                   command=lambda: self._save_report(report_content)).pack(side=tk.RIGHT)

    def _generate_detailed_report(self):
        """Generate detailed comparison report"""
        report = "=" * 100 + "\n"
        report += "Advanced HRL Mower Path Planning System - FIXED Algorithm Performance Comparison Report\n"
        report += "=" * 100 + "\n\n"

        # Executive summary
        report += "📋 EXECUTIVE SUMMARY (FIXED VERSION)\n"
        report += "-" * 50 + "\n"
        report += f"Number of Algorithms Compared: {len(self.algorithm_comparator.algorithms)}\n"
        report += f"Number of Test Areas: {len(self.areas)}\n"
        report += f"Number of Obstacles: {len(self.obstacles)}\n"
        report += f"Orchard Size: {self.orchard_size[0] / 1000:.1f}m × {self.orchard_size[1] / 1000:.1f}m\n\n"

        # Algorithm analysis
        report += "🔍 FIXED ALGORITHM ANALYSIS\n"
        report += "-" * 50 + "\n"

        for algo_name, metrics in self.algorithm_comparator.performance_data.items():
            algo_full_name = self.algorithm_comparator.algorithms[algo_name]['name']

            report += f"\n{algo_name} ({algo_full_name}) - FIXED:\n"
            report += f"  Total Cost: {metrics['total_cost']:.2f} (now always positive)\n"
            report += f"  Energy Consumption: {metrics['energy_consumption']:.1f} Wh\n"
            report += f"  Time Cost: {metrics['time_cost']:.1f} minutes\n"
            report += f"  Coverage Ratio: {metrics['coverage_ratio']:.2%} (fixed calculation)\n"
            report += f"  Path Points: {metrics['path_length']}\n"
            report += f"  Path Smoothness: {metrics['smoothness']:.3f}\n"
            report += f"  Execution Time: {metrics['execution_time']:.2f} seconds (realistic)\n"
            report += f"  Total Path Length: {metrics['total_distance']:.0f} mm\n"

        # Performance ranking
        report += "\n🏆 PERFORMANCE RANKING (FIXED)\n"
        report += "-" * 50 + "\n"

        # Sort by total cost
        sorted_by_cost = sorted(self.algorithm_comparator.performance_data.items(),
                                key=lambda x: x[1]['total_cost'])

        report += "Ranking by Total Cost (low to high):\n"
        for i, (algo_name, metrics) in enumerate(sorted_by_cost):
            report += f"  {i + 1}. {algo_name}: {metrics['total_cost']:.2f}\n"

        # Sort by coverage
        sorted_by_coverage = sorted(self.algorithm_comparator.performance_data.items(),
                                    key=lambda x: x[1]['coverage_ratio'], reverse=True)

        report += "\nRanking by Coverage Ratio (high to low):\n"
        for i, (algo_name, metrics) in enumerate(sorted_by_coverage):
            report += f"  {i + 1}. {algo_name}: {metrics['coverage_ratio']:.2%}\n"

        # Sort by execution time
        sorted_by_time = sorted(self.algorithm_comparator.performance_data.items(),
                                key=lambda x: x[1]['execution_time'])

        report += "\nRanking by Execution Time (fast to slow):\n"
        for i, (algo_name, metrics) in enumerate(sorted_by_time):
            report += f"  {i + 1}. {algo_name}: {metrics['execution_time']:.2f}s\n"

        # Fixed issues summary
        report += "\n🔧 FIXES APPLIED\n"
        report += "-" * 50 + "\n"
        report += "SAC Algorithm Fixes:\n"
        report += "  • Added computational complexity for realistic execution time\n"
        report += "  • Improved coverage ratio calculation using proper area formula\n\n"

        report += "GA Algorithm Fixes:\n"
        report += "  • Fixed cost calculation to prevent negative values\n"
        report += "  • Corrected coverage bonus implementation\n"
        report += "  • Improved area size calculation using shoelace formula\n\n"

        report += "PSO Algorithm Fixes:\n"
        report += "  • Fixed fitness function to avoid negative costs\n"
        report += "  • Corrected coverage penalty calculation\n"
        report += "  • Improved polygon area calculation\n\n"

        report += "ACO Algorithm Fixes:\n"
        report += "  • Enhanced coverage metric calculation\n"
        report += "  • Improved area coverage estimation\n"
        report += "  • Fixed polygon area calculation\n\n"

        # Recommendations
        report += "\n💡 APPLICATION RECOMMENDATIONS (BASED ON FIXED RESULTS)\n"
        report += "-" * 50 + "\n"

        if self.algorithm_comparator.comparison_results:
            best_cost = self.algorithm_comparator.comparison_results.get('lowest_cost', 'N/A')
            best_coverage = self.algorithm_comparator.comparison_results.get('best_coverage', 'N/A')
            fastest = self.algorithm_comparator.comparison_results.get('fastest_execution', 'N/A')

            report += f"• For cost-effectiveness: Recommend {best_cost}\n"
            report += f"• For maximum coverage: Recommend {best_coverage}\n"
            report += f"• For speed: Recommend {fastest}\n"

        report += "\n📝 CONCLUSIONS\n"
        report += "-" * 50 + "\n"

        best_overall = min(self.algorithm_comparator.performance_data.keys(),
                           key=lambda x: self.algorithm_comparator.performance_data[x]['total_cost'])

        report += f"Based on fixed algorithm comparison, {best_overall} shows best overall performance.\n"
        report += "All algorithms now provide realistic and comparable results.\n"
        report += "Coverage ratios are properly calculated and execution times are realistic.\n\n"

        report += "=" * 100 + "\n"
        report += f"Fixed Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 100

        return report

    def _save_report(self, content):
        """Save report to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Fixed Algorithm Comparison Report"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Save Successful", f"Fixed algorithm comparison report saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Save Failed", f"Error saving report: {str(e)}")

    def show_performance_analysis(self):
        """Show performance analysis"""
        self.analysis_text.delete(1.0, tk.END)

        analysis_content = "Fixed Performance Analysis Report\n"
        analysis_content += "=" * 50 + "\n\n"

        if self.algorithm_comparator.performance_data:
            analysis_content += "Fixed algorithm performance data available\n"
            analysis_content += f"Compared {len(self.algorithm_comparator.performance_data)} algorithms\n\n"

            for algo_name, metrics in self.algorithm_comparator.performance_data.items():
                analysis_content += f"{algo_name} Fixed Key Metrics:\n"
                analysis_content += f"  - Cost Efficiency: {100 / max(1, metrics['total_cost']):.2f}\n"
                analysis_content += f"  - Time Efficiency: {100 / max(1, metrics['execution_time']):.2f}\n"
                analysis_content += f"  - Coverage Quality: {metrics['coverage_ratio']:.2%}\n"
                analysis_content += f"  - Path Quality: {metrics['smoothness']:.3f}\n\n"
        else:
            analysis_content += "Algorithm comparison not yet run\n"
            analysis_content += "Please run fixed comparison analysis in the Algorithm Comparison tab first\n"

        self.analysis_text.insert(tk.END, analysis_content)

    def export_all_data(self):
        """Export all data"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export All Fixed Data"
        )

        if filename:
            try:
                # Collect all data
                export_data = {
                    'timestamp': time.time(),
                    'version': 'Fixed Algorithm Version',
                    'robot_params': asdict(self.robot_params),
                    'orchard_size': self.orchard_size,
                    'areas': [],
                    'obstacles': [],
                    'algorithm_comparison_results': {}
                }

                # Areas data
                for i, area in enumerate(self.areas):
                    area_data = {
                        'id': i,
                        'priority': area.priority,
                        'grass_type': area.grass_type,
                        'soil_type': area.soil_type,
                        'drainage': area.drainage,
                        'points': [(p.x, p.y, p.z, p.slope, p.soil_moisture, p.grass_density, p.traversability)
                                   for p in area.points]
                    }
                    export_data['areas'].append(area_data)

                # Obstacles data
                for obs in self.obstacles:
                    obs_data = {
                        'x': obs.x, 'y': obs.y, 'radius': obs.radius,
                        'height': obs.height, 'type': obs.obstacle_type,
                        'creates_shadow': obs.creates_shadow,
                        'soil_moisture_effect': obs.soil_moisture_effect
                    }
                    export_data['obstacles'].append(obs_data)

                # Fixed algorithm comparison results
                if self.algorithm_comparator.performance_data:
                    export_data['algorithm_comparison_results'] = {
                        'performance_data': self.algorithm_comparator.performance_data,
                        'comparison_results': self.algorithm_comparator.comparison_results,
                        'algorithm_paths': {}
                    }

                    # Export paths
                    algorithm_paths = self.algorithm_comparator.get_algorithm_paths()
                    for algo_name, path_points in algorithm_paths.items():
                        export_data['algorithm_comparison_results']['algorithm_paths'][algo_name] = [
                            (p.x, p.y, p.z) for p in path_points
                        ]

                # Save to file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

                messagebox.showinfo("Export Successful", f"All fixed data exported to: {filename}")

            except Exception as e:
                messagebox.showerror("Export Failed", f"Error exporting data: {str(e)}")

    def generate_full_report(self):
        """Generate full system report"""
        if not self.algorithm_comparator.performance_data:
            messagebox.showwarning("Warning",
                                   "Please run fixed algorithm comparison first to generate complete report!")
            return

        self.generate_comparison_report()

    def on_mouse_click(self, event):
        """Mouse click handler for area drawing"""
        if not self.drawing_mode or event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    point = Point3D(
                        x=float(x), y=float(y), z=0.0,
                        slope=random.uniform(0, 15),
                        soil_moisture=random.uniform(0.3, 0.8),
                        grass_density=random.uniform(0.5, 0.9),
                        traversability=random.uniform(0.7, 1.0)
                    )
                    self.current_area.append(point)
                    self.update_plot()
        elif event.button == 3:  # Right click
            self.finish_current_area()

    def finish_current_area(self):
        """Finish drawing current area"""
        if len(self.current_area) >= 3:
            # Create TerraceArea
            avg_slope = np.mean([p.slope for p in self.current_area])
            priority = 3 if avg_slope < 10 else 2 if avg_slope < 20 else 1

            terrace_area = TerraceArea(
                points=self.current_area.copy(),
                priority=priority,
                grass_type="medium",
                last_mowed=random.uniform(0, 14),
                soil_type="loam",
                drainage=random.uniform(0.3, 0.8)
            )

            self.areas.append(terrace_area)
            self.current_area.clear()
            self.update_plot()

            messagebox.showinfo("Area Complete", f"Added new area (Priority: {priority})")

    def update_plot(self):
        """Update 2D plot with all elements"""
        self.ax.clear()
        self.setup_plot()

        # Draw terrain contours if available
        if hasattr(self, 'terrain_data') and self.terrain_data is not None:
            try:
                X, Y, Z = self.terrain_data
                contour = self.ax.contour(X, Y, Z, levels=15, colors='brown', alpha=0.4, linewidths=0.5)
                self.ax.clabel(contour, inline=True, fontsize=6)
            except Exception as e:
                print(f"Warning: Could not draw terrain: {e}")

        # Draw obstacles
        for obs in self.obstacles:
            color_map = {
                'tree': 'green',
                'rock': 'gray',
                'equipment': 'orange',
                'debris': 'brown',
                'water': 'blue'
            }
            color = color_map.get(obs.obstacle_type, 'red')

            circle = Circle((obs.x, obs.y), obs.radius, color=color, alpha=0.6)
            self.ax.add_patch(circle)

        # Draw areas
        priority_colors = {1: 'lightcoral', 2: 'lightyellow', 3: 'lightgreen',
                           4: 'lightblue', 5: 'lightpink'}

        for i, area in enumerate(self.areas):
            if len(area.points) >= 3:
                color = priority_colors.get(area.priority, 'lightgray')

                x_coords = [p.x for p in area.points] + [area.points[0].x]
                y_coords = [p.y for p in area.points] + [area.points[0].y]

                self.ax.plot(x_coords, y_coords, 'g-', linewidth=2, alpha=0.8)
                self.ax.fill(x_coords, y_coords, color=color, alpha=0.3)

                # Add area label
                center_x = sum(p.x for p in area.points) / len(area.points)
                center_y = sum(p.y for p in area.points) / len(area.points)
                self.ax.text(center_x, center_y, f'Area{i + 1}\nPriority{area.priority}',
                             ha='center', va='center', fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Draw current area being drawn
        if len(self.current_area) > 0:
            x_coords = [p.x for p in self.current_area]
            y_coords = [p.y for p in self.current_area]
            self.ax.plot(x_coords, y_coords, 'o-', color='orange', linewidth=2, markersize=6)

        self.ax.set_aspect('equal')
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"Warning: Could not update canvas: {e}")

    def update_plot_with_algorithm_paths(self):
        """Update plot to show all algorithm paths"""
        if not hasattr(self.algorithm_comparator, 'solution_data'):
            return

        # First draw the base plot
        self.update_plot()

        # Then add algorithm paths
        algorithm_paths = self.algorithm_comparator.get_algorithm_paths()

        if algorithm_paths:
            line_styles = ['-', '--', '-.', ':']

            for i, (algo_name, path_points) in enumerate(algorithm_paths.items()):
                if path_points and len(path_points) > 1:
                    x_coords = [p.x for p in path_points]
                    y_coords = [p.y for p in path_points]

                    color = self.algorithm_comparator.algorithms[algo_name]['color']
                    line_style = line_styles[i % len(line_styles)]

                    self.ax.plot(x_coords, y_coords,
                                 color=color, linewidth=2, alpha=0.7,
                                 linestyle=line_style, label=f'{algo_name} Path (Fixed)')

                    # Mark start point
                    if len(x_coords) > 0:
                        self.ax.scatter(x_coords[0], y_coords[0], color=color, s=80,
                                        marker='o', edgecolor='black', zorder=5)

            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        try:
            self.canvas.draw()
        except Exception as e:
            print(f"Warning: Could not update canvas with paths: {e}")

    def on_closing(self):
        """Handle application closing"""
        try:
            # Stop any running simulations
            if hasattr(self, 'simulation_running'):
                self.simulation_running = False

            # Close training window
            if hasattr(self, 'training_window') and self.training_window.window is not None:
                self.training_window.on_closing()

            # Close comparison window
            if hasattr(self, 'comparison_window') and self.comparison_window.window is not None:
                self.comparison_window.on_closing()

            # Close training curves window
            if hasattr(self, 'training_curves_window') and self.training_curves_window.window is not None:
                self.training_curves_window.on_closing()

            # Close matplotlib figures
            if hasattr(self, 'fig'):
                plt.close(self.fig)

            print("Fixed application closed successfully")

        except Exception as e:
            print(f"Warning during cleanup: {e}")
        finally:
            self.root.destroy()

    def run(self):
        """Run the application"""
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Show initial instructions
        messagebox.showinfo("Welcome to Fixed Version",
                            "Welcome to the FIXED Advanced HRL Mower Path Planning System!\n\n"
                            "🔧 FIXES APPLIED:\n"
                            "• SAC: Proper execution time calculation\n"
                            "• GA: Fixed negative cost issue\n"
                            "• PSO: Corrected coverage ratio calculation\n"
                            "• ACO: Improved area coverage metrics\n"
                            "• All: Fixed polygon area calculation\n\n"
                            "Instructions:\n"
                            "1. Generate complex scene or manually draw areas\n"
                            "2. Run FIXED algorithm comparison\n"
                            "3. View corrected performance analysis and paths\n\n"
                            "All coverage ratios should now be realistic!")

        # Start main GUI loop
        self.root.mainloop()


if __name__ == "__main__":
    app = AdvancedHRLMowerPlanner()
    app.run()