import os, numpy, pygame, gymnasium
from game_environment import GameEnvironment
from stable_baselines3 import PPO
from PIL import Image

BACKGROUND_COLOR = (255,229,204)
BLACK = (0,0,0)
DISPLAY_SHAPE = (480, 480)
FPS = 24
SEED = 23

recording = True
frames = []

pygame.init()

clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode(DISPLAY_SHAPE)
pygame.display.set_caption('Bouncing Balls')

env = GameEnvironment( DISPLAY_SHAPE, 1.0/float(FPS) )
env.reset(SEED)

model = PPO.load("ppo_bouncing_balls_latest", env=env)
for episode_num in range(5):

    obs, info = env.reset()
    episode_reward = 0

    episode_over = False
    while not episode_over:

        clock.tick(FPS) # generate new frame

        gameDisplay.fill(BACKGROUND_COLOR)

        env.render(gameDisplay)

        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        episode_over = terminated or truncated

        font = pygame.font.SysFont(None, 18)
        text = font.render(f"Score: {episode_reward:.2f}", True, BLACK)
        gameDisplay.blit(text,(DISPLAY_SHAPE[0]/3,60))

        # Update Display
        pygame.display.update()

        if recording:
            pygame_frame = pygame.surfarray.array3d(gameDisplay)
            pygame_frame = pygame_frame.transpose([1, 0, 2])  # Swap axes to (height, width, channels)
            pil_image = Image.fromarray(pygame_frame)
            frames.append(pil_image)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    gameDisplay.fill(BACKGROUND_COLOR)

    font = pygame.font.SysFont(None, 42)
    text = font.render("GAME OVER", True, BLACK)
    gameDisplay.blit(text,(DISPLAY_SHAPE[0]/3,DISPLAY_SHAPE[1]/3))

    pygame.display.update()

    print(f"Episode {episode_num} ended with score: {episode_reward:.2f} in {env.step_counter} steps.")

    pygame.time.delay(3000)

if recording and len(frames) > 0:
    frames[0].save(
        'animation.gif', 
        save_all = True, 
        append_images = frames[1:], 
        duration = 1000.0/float(FPS),  # milliseconds per frame
        loop = 0 # 0 = infinite loop
    )  
    print("GIF saved as animation.gif")

env.close()

pygame.quit()
quit()
