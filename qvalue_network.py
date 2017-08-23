import tensorflow as tf

class QValueNetwork(object):
    """
    Input to the network is the state s and action a, output is Q(s,a)
    """
    def __init__(self, sess, hero_state_dim, balls_state_shape, action_dim, learning_rate, tau):
        self.sess = sess
        self.hero_states_dim = hero_state_dim
        self.ball_states_shape = balls_state_shape
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Define placeholders
        self.hero_states = tf.placeholder( tf.float32, shape=[None,self.hero_states_dim] )
        self.ball_states = tf.placeholder( tf.float32, shape=[None,self.ball_states_shape[0],self.ball_states_shape[1]] )
        self.actions = tf.placeholder( tf.float32, shape=[None, self.a_dim] )
        self.qvalues = tf.placeholder( tf.float32, shape=[None, 1] )

        # Build neural network in Tensorflow
        self.__build_graph()

        # Public Methods
        self.pred_best_actions = tf.one_hot( tf.argmax( self.h4, 1), self.a_dim )
        self.pred_qvalues = tf.diag_part( tf.matmul( self.h4, tf.transpose(self.actions) ) )
        self.pred_max_qvalues = tf.diag_part( tf.matmul( self.h4, tf.transpose(self.pred_best_actions) ) ) # Apply model best actions to Q-values of target network
        self.update_target_network_params = [ self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1.0 - self.tau)) for i in range(len(self.target_network_params)) ]

        # Define loss and optimization Op
        self.loss = tf.reduce_mean( tf.square(self.pred_qvalues - self.qvalues) )
        self.optimize = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        
        # Create Tensorboard Summaries
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("f0", self.f0)
        tf.summary.histogram("f1", self.f1)
        tf.summary.histogram("w1", self.w1)
        tf.summary.histogram("w2", self.w2)
        tf.summary.histogram("w3", self.w3)
        tf.summary.histogram("b0", self.b0)
        tf.summary.histogram("b1", self.b1)
        tf.summary.histogram("b2", self.b2)
        tf.summary.histogram("b3", self.b3)
        self.merged_summary = tf.summary.merge_all()

    def __build_network(self, scope_name):
        
        num_filters = 60
        
        def max_channel( conv_data ):
            return tf.reduce_max( conv_data, axis=3 )

        with tf.name_scope(scope_name):

            f0 = tf.Variable( tf.random_normal(shape=[1, 1, self.ball_states_shape[1], num_filters], stddev=1.0) )
            f1 = tf.Variable( tf.random_normal(shape=[1, 1, num_filters, num_filters // 2], stddev=1.0) )
            w1 = tf.Variable( tf.random_normal(shape=[self.ball_states_shape[0], self.a_dim], stddev=1.0) )

            w2 = tf.Variable( tf.random_normal(shape=[self.hero_states_dim, 2 * self.a_dim], stddev=1.0) )
            w3 = tf.Variable( tf.random_normal(shape=[2 * self.a_dim, self.a_dim], stddev=1.0) )

            b0 = tf.Variable( tf.constant(0.1, shape=[num_filters]) )
            b1 = tf.Variable( tf.constant(0.1, shape=[num_filters // 2]) )
            b2 = tf.Variable( tf.constant(0.1, shape=[2 * self.a_dim]) )
            b3 = tf.Variable( tf.constant(0.1, shape=[self.a_dim]) )

            conv_input = tf.reshape( self.ball_states, [-1,self.ball_states_shape[0],1,self.ball_states_shape[1]] )
            h0 = tf.nn.relu( tf.nn.conv2d( conv_input, f0, strides=[1,1,1,1], padding='SAME') + b0 )
            h1 = tf.nn.relu( tf.nn.conv2d( h0, f1, strides=[1,1,1,1], padding='SAME') + b1 )
            h2 = tf.reshape( max_channel( h1 ), [-1, self.ball_states_shape[0]] )

            h3 = tf.nn.relu( tf.matmul( self.hero_states, w2 ) + b2 ) 
            h4 = tf.matmul( h2, w1 ) + tf.matmul( h3, w3 ) + b3

        return f0, f1, w1, w2, w3, b0, b1, b2, b3, h0, h1, h2, h3, h4

    def __build_graph(self):

        # Create Q-Value network
        self.f0, self.f1, self.w1, self.w2, self.w3, self.b0, self.b1, self.b2, self.b3, self.h0, self.h1, self.h2, self.h3, self.h4 = self.__build_network("model")
        self.network_params = tf.trainable_variables()

        # Create Target Q-Value Network
        self.target_f0, self.target_f1, self.target_w1, self.target_w2, self.target_w3, self.target_b0, self.target_b1, self.target_b2, self.target_b3, self.target_h0, self.target_h1, self.target_h2, self.target_h3, self.target_h4 = self.__build_network("target")
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

    def train(self, hero_states, ball_states, actions, qvalues):
        return self.sess.run([self.loss, self.merged_summary, self.optimize], feed_dict={
            self.hero_states: hero_states,
            self.ball_states: ball_states,
            self.qvalues: qvalues,
            self.actions: actions
        })

    def best_actions(self, hero_states, ball_states):
        return self.sess.run(self.pred_best_actions, feed_dict={
            self.hero_states: hero_states,
            self.ball_states: ball_states
        })

    def max_qvalues(self, hero_states, ball_states): # Double Q-Learning principle
        return self.sess.run(self.pred_max_qvalues, feed_dict={
            self.hero_states: hero_states,
            self.ball_states: ball_states
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)