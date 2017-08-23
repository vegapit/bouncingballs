import os
import pygame
import numpy as np
import tensorflow as tf
from game_environment import GameEnvironment
from qvalue_network import QValueNetwork

WHITE = (255,255,255)
BLACK = (0,0,0)
DISPLAY_SHAPE = (480, 480)
FPS = 60

MODEL_FILE = './saved/bouncing-balls.ckpt'
LEARNING_RATE = 0.0001
TAU = 0.01

with tf.Session() as sess:

    pygame.init()

    clock = pygame.time.Clock()
    gameDisplay = pygame.display.set_mode(DISPLAY_SHAPE)
    pygame.display.set_caption('Bouncing Balls')

    env = GameEnvironment(DISPLAY_SHAPE,1.0/float(FPS))

    hero_state_dim = 2
    balls_state_shape = (10,5)
    action_dim = 9

    qvalue_network = QValueNetwork(sess, hero_state_dim, balls_state_shape, action_dim, LEARNING_RATE, TAU)
    
    saver = tf.train.Saver(max_to_keep=1)
    saver.restore(sess, MODEL_FILE)

    while True:

        s = env.reset()
        score, done = 0.0, False

        # Loop until terminal state
        while not done:

            clock.tick(FPS) # generate new frame

            gameDisplay.fill(WHITE)

            env.render(gameDisplay)

            a = qvalue_network.best_actions( np.expand_dims(s[0],axis=0), np.expand_dims(s[1], axis=0) ).ravel()

            action_index = np.argmax( a )

            s, reward, done = env.step(action_index)

            if (reward != 0.0):
                score += reward

            font = pygame.font.SysFont(None, 18)
            text = font.render("Score: %.2f" % score, True, BLACK)
            gameDisplay.blit(text,(DISPLAY_SHAPE[0]/3,60))

            # Update Display
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        gameDisplay.fill(WHITE)

        font = pygame.font.SysFont(None, 42)
        text = font.render("GAME OVER", True, BLACK)
        gameDisplay.blit(text,(DISPLAY_SHAPE[0]/3,DISPLAY_SHAPE[1]/3))

        pygame.display.update()

        pygame.time.delay(3000)

    pygame.quit()
    quit()
