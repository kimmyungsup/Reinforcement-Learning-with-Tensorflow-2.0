import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')
num_action = 1

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.layer_a1 = Dense(64, activation='relu')
        self.layer_a2 = Dense(64, activation='relu')
        self.logits = Dense(num_action, activation='tanh')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        logits = self.logits(layer_a2)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.layer_c1 = Dense(64, activation='relu')
        self.layer_c2 = Dense(64, activation='relu')
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
        self.en = 0.001

        self.actor_model = ActorModel()
        self.actor_opt = optimizers.Adam(lr=self.lr, )

        self.critic_model = CriticModel()
        self.critic_opt = optimizers.Adam(lr=self.lr2, )

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)

    def actor_loss(self, states):
        actions = self.actor_model(tf.convert_to_tensor(states, dtype=tf.float32))
        state_actions = tf.concat([states, actions], 1)
        state_actions = tf.cast(state_actions, tf.float32)
        Q_values = self.critic_model(tf.convert_to_tensor(state_actions, dtype=tf.float32))
        loss = -tf.reduce_mean(Q_values)
        return loss

    def critic_loss(self, states, actions, rewards, dones):
        state_actions = tf.concat([states, actions], 1)
        state_actions = tf.cast(state_actions, tf.float32)
        last = state_actions[-1]
        if dones[-1] == True :
            reward_sum = 0
        else :
            reward_sum = self.critic_model(tf.convert_to_tensor(last[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        #discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32)
        values = self.critic_model(tf.convert_to_tensor(state_actions, dtype=tf.float32))
        error = tf.square(values - discounted_rewards)*0.5
        error = tf.reduce_mean(error)
        return error

    #@tf.function
    def train(self, states, actions, rewards, dones):
        states = tf.cast(states, tf.float32)
        with tf.GradientTape() as tape_critic:
            critic_loss = self.critic_loss(states, actions, rewards, dones)

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(tf.convert_to_tensor(states, dtype=tf.float32))
            state_actions = tf.concat([states, actions], 1)
            state_actions = tf.cast(state_actions, tf.float32)
            Q_values = self.critic_model(tf.convert_to_tensor(state_actions, dtype=tf.float32))
            actor_loss = -tf.reduce_mean(Q_values)
            #actor_loss = self.actor_loss(states)

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # self.train_loss(tf.reduce_mean(actor_loss))
        # self.train_loss_c(tf.reduce_mean(critic_loss))
        self.train_loss = tf.reduce_mean(actor_loss)
        self.train_loss_c = tf.reduce_mean(critic_loss)


    def run(self):

        t_end = 200
        epi = 100000
        train_size = 20

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        state = env.reset()
        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                action = self.actor_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                action = np.array(action)[0]
                next_state, reward, done, _ = env.step(action)
                reward = next_state[0]
                #env.render()
                if t == t_end :
                    done = True
                if t < t_end and done :
                    reward = -1

                total_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state


                if len(states) == train_size or done :
                    self.train(states, actions, rewards, dones)
                    states = []
                    actions = []
                    rewards = []
                    next_states = []
                    dones = []

                if done:
                    self.reward_board(total_reward)
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    with self.train_summary_writer.as_default():
                        # tf.summary.scalar('reward', self.reward_board.result(), step=e)
                        tf.summary.scalar('actor_loss', self.train_loss, step=e)
                        tf.summary.scalar('critic_loss', self.train_loss_c, step=e)
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    ActorCritic = ActorCriticTrain()
    ActorCritic.run()

