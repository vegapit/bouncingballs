import numpy as np
import pygame
from game_environment import GameEnvironment
from experience_replay import ExperienceReplay

# Experience Replay parameters
EXP_REPLAY_FILE = 'exp_replay.pkl'
BUFFER_SIZE = 12000
MINIBATCH_SIZE = 300
SAVE_EXPERIENCE = False

WHITE = (255,255,255)
BLACK = (0,0,0)

pygame.init()

DISPLAY_SHAPE = (480,480)
FPS = 60

clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode(DISPLAY_SHAPE)
pygame.display.set_caption('Bouncing Balls')
pygame.key.set_repeat(1, 1)

env = GameEnvironment(DISPLAY_SHAPE,1.0/float(FPS))

def action_vector(a):
    res = np.zeros(9)
    res[int(a)] = 1.0
    return res

# Define Experience Replay
if SAVE_EXPERIENCE:
    er = ExperienceReplay.load(EXP_REPLAY_FILE)
    if er == None:
        er = ExperienceReplay(BUFFER_SIZE)

def gameover(hero_score):

    gameDisplay.fill(WHITE)

    font = pygame.font.SysFont(None, 42)
    text = font.render("GAME OVER", True, BLACK)
    gameDisplay.blit(text,(DISPLAY_SHAPE[0]/3,DISPLAY_SHAPE[1]/3))

    pygame.display.update()

    pygame.time.delay(3000)

def gameLoop():

    s = env.reset()
    score, terminal = 0.0, False
    
    # Loop until terminal state
    while not terminal:

        clock.tick(FPS) # generate new frame

        gameDisplay.fill(WHITE)

        env.render(gameDisplay)

        a = 0
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP8:
                    a = 1
                elif event.key == pygame.K_KP9:
                    a = 2
                elif event.key == pygame.K_KP6:
                    a = 3
                elif event.key == pygame.K_KP3:
                    a = 4
                elif event.key == pygame.K_KP2:
                    a = 5
                elif event.key == pygame.K_KP1:
                    a = 6
                elif event.key == pygame.K_KP4:
                    a = 7
                elif event.key == pygame.K_KP7:
                    a = 8
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        s2, r, terminal = env.step(a)

        # store experience
        if SAVE_EXPERIENCE:
            if np.abs(r) > 0.0:
                er.add_experience(s, a, r, terminal, s2)
            else:
                if np.random.random() < 0.0018:
                    er.add_experience(s, a, r, terminal, s2)

        s = s2
        score += r
        
        font = pygame.font.SysFont(None, 18)
        text = font.render("Score: %.2f" % score, True, BLACK)
        gameDisplay.blit(text,(DISPLAY_SHAPE[0]/2-30,60))

        # Update Display
        pygame.display.update()

    gameover(score)

while True:
    gameLoop()
    if SAVE_EXPERIENCE:
        er.save(EXP_REPLAY_FILE)

pygame.quit()
quit()
