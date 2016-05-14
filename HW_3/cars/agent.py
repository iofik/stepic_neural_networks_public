import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np
import pygame.key

from cars.utils import Action
from learning_algorithms.network import Network


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass

    @abstractmethod
    def learn(self):
        pass

class SimpleCarAgent(Agent):
    # One NN for all:
    _rays = 5
    _neural_net = Network(
            [
                _rays + 4,
                # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                # например, (self.rays + 4) * 2 или просто число
                (_rays + 4) * 2,
                4,
                1
                ],
            output_function=lambda x: x,
            output_derivative=lambda x: 1,
            l1 = 0,
            l2 = 0,
            )

    def __init__(self, history_data=int(50000)):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self.allow_kb_control = False
        self.kb_control = False
        self._rays = SimpleCarAgent._rays
        self.neural_net = SimpleCarAgent._neural_net
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0
        self.eta = 0.05
        self.prev_error = np.infty
        self.train_every = 50  # сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        self.reward_depth = 7 # на какую глубину по времени распространяется полученная награда

    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без него не пройдёт нормально eval
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def process_kb_input(self, velocity):
        if not self.allow_kb_control:
            return

        kb_steer = 0
        kb_accel = 0
        evs = pygame.key.get_pressed()

        if evs[275]:
            kb_accel = 0.75
            kb_steer = -1
        elif evs[276]:
            kb_accel = 0.75
            kb_steer = 1
        if evs[273]:
            kb_accel = 0.75
        elif evs[274]:
            kb_accel = -0.75

        if kb_accel != 0:
            self.kb_control = True
        elif evs[97]: # A
            self.kb_control = False
        if velocity > 0.2 and random.random() < 0.2*velocity:
            kb_accel = -0.75

        return Action(kb_steer, kb_accel) if self.kb_control else None

    def choose_action(self, sensor_info):
        best_action = self.process_kb_input(sensor_info[0])

        if not best_action:
            # хотим предсказать награду за все действия, доступные из текущего состояния
            rewards_to_controls_map = {}
            # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
            for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
                for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                    action = Action(steering, acceleration)
                    agent_vector_representation = np.append(sensor_info, action)
                    agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                    predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                    rewards_to_controls_map[predicted_reward] = action

            # ищем действие, которое обещает максимальную награду
            rewards = list(rewards_to_controls_map.keys())
            highest_reward = max(rewards)
            best_action = rewards_to_controls_map[highest_reward]

            # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
            # рандомное действие
            if (not self.evaluate_mode) and (random.random() < 0.05):
                highest_reward = rewards[np.random.choice(len(rewards))]
                best_action = rewards_to_controls_map[highest_reward]
            # следующие строки помогут вам понять, что предсказывает наша сеть
            #     print("Chosen random action w/reward: {}".format(highest_reward))
            # else:
            #     print("Chosen action w/reward: {}".format(highest_reward))

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром

        return best_action

    def receive_feedback(self, reward):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < self.reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= self.train_every выполнялось
        if not self.evaluate_mode and not self.kb_control and (len(self.reward_history) >= self.train_every) and not (self.step % self.train_every):
            self.learn()

    def learn(self, final=False):
        X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
        y_train = self.reward_history
        train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
        epochs = 30 if final else 15
        self.neural_net.SGD(training_data=train_data, epochs=epochs, mini_batch_size=self.train_every, eta=self.eta, verbose=final)
        y_hat = self.neural_net.feedforward(X_train.transpose())
        d = y_train - y_hat
        error = np.sum(d*d)/d.shape[1]
        if error > self.prev_error:
            self.eta /= 1.1
        self.prev_error = error
        print("Step: %4d      Error function: %9.6f" % (self.step, error))
