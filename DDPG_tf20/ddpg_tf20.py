import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import random
from collections import deque
import gym

env = gym.make('Pendulum-v0')
num_action = 1
state_size = env.observation_space.shape[0]

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.layer_a1 = Dense(128, activation='relu')
        self.layer_a2 = Dense(128, activation='relu')
        self.logits = Dense(num_action, activation='tanh')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        logits = self.logits(layer_a2)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.layer_c1 = Dense(128, activation='relu')
        self.layer_c2 = Dense(128, activation='relu')
        self.value = Dense(1)

    def call(self, state_action):
        layer_c1 = self.layer_c1(state_action)
        layer_c2 = self.layer_c2(layer_c1)
        value = self.value(layer_c2)
        return value


class ActorCriticTrain:
    def __init__(self):
        # hyper parameters
        self.lr =0.001
        self.lr2 = 0.001
        self.df = 0.99
        self.tau = 0.001
        self.N_std = 0.1

        self.actor_model = ActorModel()
        self.actor_target = ActorModel()
        self.actor_opt = optimizers.Adam(lr=self.lr, )

        self.critic_model = CriticModel()
        self.critic_target = CriticModel()
        self.critic_opt = optimizers.Adam(lr=self.lr2, )

        self.critic_target.set_weights(self.critic_model.get_weights())
        self.actor_target.set_weights(self.actor_model.get_weights())

        self.batch_size = 256
        self.train_start = 1000
        self.state_size = state_size

        self.memory = deque(maxlen=10000)

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)

    def update_target(self, target_weights, source_weights):
        use_locking = False
        target_variables = target_weights
        source_variables = source_weights

        update_ops = []
        for target_var, source_var in zip(target_variables, source_variables) :
            update_ops.append(target_var.assign(self.tau * source_var + (1.0 - self.tau) * target_var, use_locking))

        return tf.group(name="update_all_variables", *update_ops)


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #@tf.function
    def train(self):

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        states = tf.cast(states, tf.float32)

        target_actions = self.actor_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))

        with tf.GradientTape() as tape_critic:
            state_actions = tf.concat([states, actions], 1)
            state_actions = tf.cast(state_actions, tf.float32)

            target_state_actions = tf.concat([next_states, target_actions], 1)
            target_state_actions = tf.cast(target_state_actions, tf.float32)

            target = self.critic_model(tf.convert_to_tensor(np.vstack(state_actions), dtype=tf.float32))
            target_val = self.critic_target(tf.convert_to_tensor(np.vstack(target_state_actions), dtype=tf.float32))

            target = np.array(target)
            target_val = np.array(target_val)

            for i in range(self.batch_size):
                next_v = np.array(target_val[i]).max()
                if dones[i]:
                    target[i] = rewards[i]
                else:
                    target[i] = rewards[i] + self.df * next_v

            values = self.critic_model(tf.convert_to_tensor(np.vstack(state_actions), dtype=tf.float32))
            error = tf.square(values - target) * 0.5
            error = tf.reduce_mean(error)

        critic_grads = tape_critic.gradient(error, self.critic_model.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(tf.convert_to_tensor(states, dtype=tf.float32))
            state_actions = tf.concat([states, actions], 1)
            state_actions = tf.cast(state_actions, tf.float32)
            Q_values = self.critic_model(tf.convert_to_tensor(state_actions, dtype=tf.float32))
            actor_loss = -tf.reduce_mean(Q_values)


        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        self.train_loss = tf.reduce_mean(actor_loss)
        self.train_loss_c = tf.reduce_mean(error)


    def run(self):

        t_end = 1000
        epi = 100000

        state = env.reset()

        for e in range(epi):
            total_reward = 0

            for t in range(t_end):
                action = self.actor_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                action = np.array(action)[0] + np.random.normal(loc=0.0, scale=self.N_std, size=[num_action])

                next_state, reward, done, _ = env.step(action)

                if e > 100 : env.render()
                #print(state[0])

                if t == t_end :
                    done = True


                total_reward += reward

                self.append_sample(state, action, reward, next_state, done)

                state = next_state


                if done :
                    if len(self.memory) >= self.train_start:
                        self.train()
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar('actor_loss', self.train_loss, step=e)
                            tf.summary.scalar('critic_loss', self.train_loss_c, step=e)
                        self.update_target(self.actor_target.weights, self.actor_model.weights)
                        self.update_target(self.critic_target.weights, self.critic_model.weights)
                    self.reward_board(total_reward)
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    ActorCritic = ActorCriticTrain()
    ActorCritic.run()

