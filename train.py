import numpy as np
import tensorflow as tf
from game_environment import GameEnvironment
from experience_replay import ExperienceReplay
from qvalue_network import QValueNetwork
import matplotlib.pyplot as plt

# ==========================
#      Game Parameters
# ==========================
DISPLAY_SHAPE = (480,480)
FPS = 60

# ==========================
#    Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 30
# Base learning rate for the QValue network
LEARNING_RATE = 1e-6
# Discount factor
GAMMA = 0.84
# Soft target update param
TAU = 1e-4
# Epsilon Greedy parameter
EPSILON_ALPHA = 0.5
EPSILON_LIMIT = 0.1
STEPS_LIMIT = 1200
EPSILON_BETA = 1.0 / STEPS_LIMIT * np.log(EPSILON_ALPHA/EPSILON_LIMIT)
# Tensorflow Parameters
MODEL_FILE = './saved/bouncing-balls.ckpt'

# ===========================
#      Model Parameters
# ===========================
# Save and Restore functionality
SAVE_AND_RESTORE = True
SAVE_TENSORBOARD = False
VISUALIZE_WEIGHTS = False
UPDATE_REPLAY = True
# Experience Replay parameters
EXP_REPLAY_FILE = 'exp_replay.pkl'
BUFFER_SIZE = 12000
MINIBATCH_SIZE = 300

with tf.Session() as sess:

    hero_state_dim = 2
    balls_state_shape = (10,5)
    action_dim = 9

    def generate_random_action():
        i = np.random.randint(0,action_dim)
        res = np.zeros(action_dim)
        res[i] = 1.0
        return res

    qvalue_network = QValueNetwork(sess, hero_state_dim, balls_state_shape, action_dim, LEARNING_RATE, TAU)

    # Define TF Saver Object
    if SAVE_AND_RESTORE:
        saver = tf.train.Saver(max_to_keep=1)
        if tf.train.latest_checkpoint('./saved/') != None:
            saver.restore(sess, MODEL_FILE)
            print("============ Model Restored ============")
        else:
            # Initialize TF Variables
            sess.run( tf.global_variables_initializer() )
    else:
        # Initialize TF Variables
        sess.run( tf.global_variables_initializer() )
    
    # Initialize target network weights
    qvalue_network.update_target_network()

    # Define Tensorboard file writer
    if SAVE_TENSORBOARD:
        writer = tf.summary.FileWriter('./saved/tensorboard/')
        writer.add_graph( sess.graph )

    # Define Experience Replay
    er = ExperienceReplay.load(EXP_REPLAY_FILE)
    if er == None:
        er = ExperienceReplay(BUFFER_SIZE)

    env = GameEnvironment(DISPLAY_SHAPE,1.0/float(FPS))

    fig, ax_list = plt.subplots(5,1)

    for i in range(MAX_EPISODES):

        s = env.reset()
        
        ep_reward = 0
        terminal = False
        num_steps = 0
        l = 1.0

        while not terminal:

            # Epsilon Greedy
            if np.random.random() < EPSILON_ALPHA * np.exp(-EPSILON_BETA * num_steps):
                a = generate_random_action()
            else:
                a = qvalue_network.best_actions( np.expand_dims(s[0],axis=0), np.expand_dims(s[1], axis=0) ).ravel()

            # Collect environment data
            s2, r, terminal = env.step( np.argmax(a) )

            # Add data to ExperienceReplay memory
            if UPDATE_REPLAY:
                if np.abs(r) > 0.0:
                    er.add_experience(s, a, r, terminal, s2)
                else:
                    if np.random.random() < 0.0018:
                        er.add_experience(s, a, r, terminal, s2)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if er.size() > MINIBATCH_SIZE:

                s_batch, a_batch, r_batch, t_batch, s2_batch = er.get_batch(MINIBATCH_SIZE)

                # Calculate Q targets for s2 based on QValue target model
                s2_batch1 = np.reshape( [elt[0].ravel() for elt in s2_batch], [-1,hero_state_dim] )
                s2_batch2 = np.reshape( [elt[1] for elt in s2_batch], [-1,balls_state_shape[0],balls_state_shape[1]] )
                target_q = qvalue_network.max_qvalues( s2_batch1, s2_batch2 )
                
                new_q = np.zeros((MINIBATCH_SIZE, 1))
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        new_q[k,0] = r_batch[k]
                    else:
                        new_q[k,0] = r_batch[k] + GAMMA * target_q[k]

                # Update qvalues given the Q targets
                s_batch1 = np.reshape( [elt[0].ravel() for elt in s_batch], [-1,hero_state_dim] )
                s_batch2 = np.reshape( [elt[1] for elt in s_batch], [-1,balls_state_shape[0],balls_state_shape[1]] )
                l, summary_str, _ = qvalue_network.train( s_batch1, s_batch2, np.reshape( a_batch, [-1, action_dim] ), new_q )

                # Update target networks
                qvalue_network.update_target_network()

            s = s2
            ep_reward += r
            num_steps += 1

        print( 'Episode: %i | Total Reward: %.1f | Total Steps: %d | Last Loss : %.3f' % (i, ep_reward, num_steps, l) )
        if SAVE_TENSORBOARD:
            writer.add_summary(summary_str, i)

        if VISUALIZE_WEIGHTS:
            ax_list[0].imshow(qvalue_network.f0.eval()[0][0], cmap='coolwarm')
            ax_list[0].set_xticks([])
            ax_list[0].set_yticks([])
            ax_list[1].imshow(qvalue_network.f1.eval()[0][0], cmap='coolwarm')
            ax_list[1].set_xticks([])
            ax_list[1].set_yticks([])
            ax_list[2].imshow(qvalue_network.w1.eval(), cmap='coolwarm')
            ax_list[2].set_xticks([])
            ax_list[2].set_yticks([])
            ax_list[3].imshow(qvalue_network.w2.eval(), cmap='coolwarm')
            ax_list[3].set_xticks([])
            ax_list[3].set_yticks([])
            ax_list[4].imshow(qvalue_network.w3.eval(), cmap='coolwarm')
            ax_list[4].set_xticks([])
            ax_list[4].set_yticks([])
            plt.draw()
            plt.pause(0.5)

    # Save Experience replay
    er.save(EXP_REPLAY_FILE)
    
    print("Experience Replay Statistics:")
    print("Size:", er.size())
    reward_history = np.array(er.rewards)
    print("Positive count:",len(reward_history[reward_history > 0]))
    print("Neutral count:",len(reward_history[reward_history == 0.0]))
    print("Negative count:",len(reward_history[reward_history < 0]))
    
    # Save calibrated model
    if SAVE_AND_RESTORE:
        save_path = saver.save(sess, MODEL_FILE)
        print("============ Model Saved ============")





 




