import numpy as np
import random

class Environment:
    def __init__(self, Ny=8, Nx=8, N_mountains=4, N_quicksands=4):
        self.Ny, self.Nx = Ny, Nx
        self.state_dim = (Ny, Nx)
        self.action_dim = (4,) 
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.end_state = (Ny - 1, Nx - 1)
        self.mountains = self._generate_obstacles(N_mountains)
        self.quicksands = self._generate_obstacles(N_quicksands)
        self.R = self._build_rewards()
        self.state = (0, 0)
        self.end_state = (Ny-1, Nx-1)
        self.path = []

    def reset(self):
        self.state = (0, 0)
        self.path = [self.state]
        return self.state

    def step(self, action):
        next_state = tuple(self.state[i] + self.action_coords[action][i] for i in range(2))
        if next_state in self.mountains or not (0 <= next_state[0] < self.Ny and 0 <= next_state[1] < self.Nx):
            next_state = self.state
        
        self.state = next_state
        self.path.append(self.state)
        reward = self.R[self.state + (action,)]
        done = self.state == self.end_state or self.state in self.quicksands
        return self.state, reward, done

    def allowed_actions(self):
        actions_allowed = []
        y, x = self.state
        if y > 0 and (y - 1, x) not in self.mountains: actions_allowed.append(self.action_dict["up"])
        if y < self.Ny - 1 and (y + 1, x) not in self.mountains: actions_allowed.append(self.action_dict["down"])
        if x > 0 and (y, x - 1) not in self.mountains: actions_allowed.append(self.action_dict["left"])
        if x < self.Nx - 1 and (y, x + 1) not in self.mountains: actions_allowed.append(self.action_dict["right"])
        return np.array(actions_allowed, dtype=int)

    def _build_rewards(self):
        R = -0.1 * np.ones(self.state_dim + self.action_dim, dtype=float)
        for qs in self.quicksands:
            for action in range(4):
                R[qs + (action,)] = -100
        R[self.Ny - 2, self.Nx - 1, self.action_dict["down"]] = 100
        R[self.Ny - 1, self.Nx - 2, self.action_dict["right"]] = 100
        return R

    def _generate_obstacles(self, N_obstacles):
        obstacles = set()
        while len(obstacles) < N_obstacles:
            obstacle = (random.randint(1, self.Ny - 2), random.randint(1, self.Nx - 2))
            if obstacle not in obstacles and obstacle != self.end_state and obstacle != (0, 0):
                obstacles.add(obstacle)
        return list(obstacles)

    def display_path(self):
        grid = np.full(self.state_dim, '.')
        for y, x in self.mountains:
            grid[y, x] = 'M'
        for y, x in self.quicksands:
            grid[y, x] = 'Q'
        for y, x in self.path:
            if grid[y, x] == '.':
                grid[y, x] = '*'
        grid[self.end_state] = 'E'
        grid[0, 0] = 'S'
        print("Gridworld Path:")
        for row in grid:
            print(' '.join(row))

class Agent:
    def __init__(self, env):
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.beta = 0.99
        self.gamma = 0.99
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(env.allowed_actions())
        else:
            state = env.state
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma * np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for y in range(self.state_dim[0]):
            for x in range(self.state_dim[1]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        print("\nGreedy policy (y, x):")
        print(greedy_policy)
        print()

env = Environment()
agent = Agent(env)

print("\nTraining agent...\n")
N_episodes = 500
for episode in range(N_episodes):
    state = env.reset()
    while True:
        action = agent.get_action(env)
        state_next, reward, done = env.step(action)
        agent.train((state, action, state_next, reward, done))
        if done:
            break
        state = state_next
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, 0.01)
# agent.display_greedy_policy()
env.display_path()
