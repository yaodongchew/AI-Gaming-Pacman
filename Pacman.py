from collections import defaultdict
import math
import pygame
import numpy as np

# Set up the game window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800
pygame.init()
game_window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pac-Man with AI Ghosts")

# Creating the PacMan character
class PacMan(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("pacman.png").convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = WINDOW_WIDTH / 2
        self.rect.y = WINDOW_HEIGHT / 2
        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        elif keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        elif keys[pygame.K_UP]:
            self.rect.y += self.speed
        elif keys[pygame.K_DOWN]:
            self.rect.y -= self.speed

# Creating the AI-Controlled ghost characters
class Ghost(pygame.sprite.Sprite):
    def __init__(self,color):
        super().__init__()
        self.color = color
        self.image = pygame.Surface([20, 20])
        self.image.fill(color)
        self.rect = self.image.get.rect()
        self.rect.x = WINDOW_WIDTH / 2 - 10
        self.rect.y = WINDOW_HEIGHT / 2 - 10
        self.speed = 3

    def update(self, target):
        dx = target.rect.x - self.rect.x
        dy = target.rect.y - self.rect.y
        distance = math.hypot(dx,dy)
        dx, dy = dx / distance, dy / distance
        self.rect.x += dx * self.speed
        self.rect.y += dy * self.speed

# Implementing Reinforcement Learning Algorithms for ghost movement
class QGhost(Ghost):
    def __init__(self, color, learning_rate = 0.1, discount_factor = 0.9, exploration_rate = 0.1):
        super().__init__(color)
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def update(self, target):
        state = self.get_state(target)
        action = self.get_action(state)
        reward = self.get_reward(target)

        next_state = self.get_state(target)
        next_action = self.get_action(next_state)

        q_value = self.q_table[state][action]
        next_q_value = self.q_table[next_state][next_action]
        td_error = reward + self.discount_factor * next_q_value - q_value
        self.q_table[state][action] += self.learning_rate * td_error

        dx, dy = self.get_direction(action)
        self.rect.x += dx * self.speed
        self.rect.y += dy * self.speed

    def get_state(self, target):
        dx = target.rect.x - self.rect.x
        dy = target.rect.y - self.rect.y
        state = round(dx / 10), round(dy / 10)
        return state
    
    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(self.q_table[state])
        return action
    
    def get_reward(self, target):
        dx = target.rect.x - self.rect.x
        dy = target.rect.y - self.rect.y
        distance = math.hypot(dx, dy)
        reward = -distance
        return reward
    
    def get_direction(self, action):
        if action == 0:
            return 1, 0
        elif action == 1:
            return -1, 0
        elif action == 2:
            return 0, 1
        else:
            return 0, -1
        
    