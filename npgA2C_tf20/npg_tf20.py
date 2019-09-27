import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import gym

env = gym.make('CartPole-v0')
num_action = env.action_space.n

class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.layer_a1 = Dense(24, activation='relu')
        self.layer_a2 = Dense(24, activation='relu')
        self.logits = Dense(num_action, activation='softmax')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        logits = self.logits(layer_a2)
        return logits

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.layer_c1 = Dense(24, activation='relu')
        self.layer_c2 = Dense(24, activation='relu')
        self.value = Dense(1)

    def call(self, state):
        layer_c1 = self.layer_c1(state)
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
        self.actor_opt = optimizers.RMSprop(lr=self.lr, )

        self.critic_model = CriticModel()
        self.critic_opt = optimizers.RMSprop(lr=self.lr2, )

        # tensorboard
        self.log_dir = 'logs/'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)
        #self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        #self.train_loss_c = tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32)

    def actor_loss(self, states, actions, values, rewards, next_values, dones):
        policy = self.actor_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        advantages = []
        for i in range(len(states)):
            reward = np.array(rewards[i])
            value = np.array(values[i])
            next_value = np.array(next_values[i])

            if dones[i]:
                advantages.append(reward - value)
            else:
                advantages.append(reward + self.df * next_value - value)
        advantages = tf.reshape(advantages, [len(states)])
        tf.convert_to_tensor(advantages, dtype=tf.float32)

        # SparseCategoricalCrossentropy
        entropy = losses.categorical_crossentropy(policy, policy, from_logits=True)
        ce_loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        #policy_loss = ce_loss(actions, policy, sample_weight=np.array(advantages))  # same way
        log_pi = ce_loss(actions, policy)
        policy_loss = log_pi * np.array(advantages)
        policy_loss = tf.reduce_mean(policy_loss)
        log_pi = tf.reduce_mean(log_pi)
        return policy_loss - self.en * entropy, log_pi


    def critic_loss(self, states, rewards, dones):
        last_state = states[-1]
        if dones[-1] == True :
            reward_sum = 0
        else :
            reward_sum = self.critic_model(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32)
        values = self.critic_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        error = tf.square(values - discounted_rewards)*0.5
        error = tf.reduce_mean(error)
        return error

    def inverse_fisher(self, psi, actor_grads):
        for l in range(len(psi)) :
            if len(tf.shape(psi[l])) == 2 :
                psi_f = tf.reshape(psi[l], [-1])
                v_len = len(psi_f)
                psi_f = tf.reshape(psi_f, [v_len, 1])
                F = tf.matmul(psi_f, tf.transpose(psi_f))
                S, U, V = tf.linalg.svd(F)
                atol = tf.reduce_max(S) * 1e-6
                S_inv = tf.divide(1.0, S)
                S_inv = tf.where(S < atol, tf.zeros_like(S), S_inv)
                S_inv = tf.linalg.diag(S_inv)
                F_inv = tf.matmul(S_inv, tf.transpose(U))
                F_inv = tf.matmul(V, F_inv)
                actor_grads_f = tf.reshape(actor_grads[l], [-1])
                actor_grads_f = tf.reshape(actor_grads_f, [v_len, 1])
                steepest_grad = tf.matmul(F_inv, actor_grads_f)
                steepest_grad = tf.reshape(steepest_grad, [len(actor_grads[l]), len(actor_grads[l][0])])
                actor_grads[l] = steepest_grad
        return actor_grads


    def train(self, states, actions, rewards, next_states, dones):
        advantages, values, next_values = self.compute_advantages_value(states, next_states, rewards, dones)

        critic_variable = self.critic_model.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            critic_loss = self.critic_loss(states, rewards, dones)

        # gradient descent will be applied automatically by apply_gradients
        critic_grads = tape_critic.gradient(critic_loss, critic_variable)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_variable))

        # persistent=True  => multi call gradient
        actor_variable = self.actor_model.trainable_weights
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(actor_variable)
            actor_loss, log_pi = self.actor_loss(states, actions, values, rewards, next_values, dones)

        actor_grads = tape.gradient(actor_loss, actor_variable)
        psi = tape.gradient(log_pi, actor_variable)

        actor_grads = self.inverse_fisher(psi, actor_grads)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_variable))


        self.train_loss = tf.reduce_mean(actor_loss)
        self.train_loss_c = tf.reduce_mean(critic_loss)

    def compute_advantages_value(self, states, next_states, rewards, dones):
        advantages = []
        value_target = []
        values = []
        next_values = []
        for i in range(len(states)):
            state = np.array(states[i])
            next_state = np.array(next_states[i])
            value = self.critic_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            next_value = self.critic_model(tf.convert_to_tensor(next_state[None, :], dtype=tf.float32))
            reward = np.array(rewards[i])
            if dones[i] :
                advantages.append(reward - value)
                value_target.append(reward)
            else :
                advantages.append(reward + self.df * next_value - value)
                value_target.append(reward + self.df * next_value)
            values.append(value)
            next_values.append(next_value)
        advantages = np.reshape(advantages, [len(states)])
        return advantages, values, next_values

    def run(self):

        t_end = 500
        epi = 3000
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
                policy = self.actor_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
                action = np.array(action)[0]
                next_state, reward, done, _ = env.step(action)

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


                if len(states) == train_size or done:
                    self.train(states, actions, rewards, next_states, dones)
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
                        # tf.summary.scalar('actor_loss', self.train_loss.result(), step=e)
                        tf.summary.scalar('actor_loss', self.train_loss, step=e)
                        tf.summary.scalar('critic_loss', self.train_loss_c, step=e)
                        tf.summary.scalar('reward', total_reward, step=e)
                    break


if __name__ == '__main__':
    ActorCritic = ActorCriticTrain()
    ActorCritic.run()

