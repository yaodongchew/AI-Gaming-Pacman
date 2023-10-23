import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

class NNGhost(Ghost):
    def __init__(self, color):
        super().__init__(color)
        self.model = self.create_model

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(2,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    
    def update(self, target):
        state = self.get_state(target)
        action = self.get_action(state)
        reward = self.get_reward(target)

        next_state = self.get_state(target)
        next_action = self.get_action(next_state)

        q_values = self.model.predict(np.array([state]))
        q_values[0][action] = reward + 0.9 * np.max(self.model.predict(np.array([next_state])))

        self.model.fit(np.array([state]), q_values, verbose = 0)

        dx, dy = self.get_direction(action)
        self.rect.x += dx * self.speed
        self.rect.y += dy * self.speed

    def get_state(self, target):
        dx = target.rect.x - self.rect.x
        dy = target.rect.y - self.rect.y
        state = np.array([dx, dy]) / 10
        return state
    
    def get_action(self, state):
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])
    
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
        
        