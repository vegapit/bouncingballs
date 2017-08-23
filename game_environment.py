import numpy as np
from ball import Ball
from player_ball import PlayerBall

RED = (255,0,0)
GREEN = (0,102,0)
BLUE = (0,128,255)
COLORS = {'red': RED, 'green': GREEN}

def generate_random_speed_vector(minspeed, maxspeed):
    vx, vy = np.random.randint(minspeed, maxspeed+1), np.random.randint(minspeed, maxspeed+1)
    if np.random.random_sample() < 0.5:
        vx = -vx
    if np.random.random_sample() < 0.5:
        vy = -vy
    return [vx,vy]

def generate_random_starting_position(maxdistance, display_height, display_width):
    x, y = np.random.randint(8, maxdistance+1), np.random.randint(8, maxdistance+1)
    if np.random.random_sample() < 0.5:
        x = display_width - x
    if np.random.random_sample() < 0.5:
        y = display_height - y
    return [x,y]

class GameEnvironment( object ):

    def __init__(self, display_shape, dt):
        self.display_width, self.display_height = display_shape[0], display_shape[1]
        self.dt = dt
        self.motion_step = 2

    def reset(self):
        self.done = False

        #Balls
        self.ball_inventory = {'red': 4, 'green': 6}
        self.balls = []
        for key in self.ball_inventory.keys():
            for i in range(self.ball_inventory[key]):
                self.balls.append( Ball(generate_random_starting_position(20, self.display_height, self.display_width),
                                    generate_random_speed_vector(70,130), COLORS[key], (self.display_width, self.display_height)) )

        #Players balls
        self.hero_ball = PlayerBall([self.display_width/2,self.display_height/4],BLUE,(self.display_width,self.display_height))

        return self._get_state()

    def _get_state(self):

        x0 = self.hero_ball.pos[0]/float(self.display_width)
        y0 = self.hero_ball.pos[1]/float(self.display_height)

        balls_state = []

        for ball in self.balls:
            if ball.live:
                x = ball.pos[0]/float(self.display_width)
                y = ball.pos[1]/float(self.display_height)
                Vx = ball.v[0]/float(self.display_width)
                Vy = ball.v[1]/float(self.display_height)
                if ball.color == GREEN:
                    s = 0.1
                else:
                    s = -1.0
                balls_state.append([x,y,Vx,Vy,s])
            else:
                balls_state.append([0.0,0.0,0.0,0.0,0.0])
                
        return [np.array([x0,y0]), np.vstack(balls_state).reshape((len(self.balls),5))]

    def step(self, action):
        x_change, y_change = 0.0, 0.0

        # 0: No move, 1: Up, 2: Up&Right...
        if action in [8,1,2] and (self.hero_ball.pos[1] + self.motion_step > self.hero_ball.r):
            y_change = self.motion_step
        if action in [6,5,4] and (self.hero_ball.pos[1] + self.motion_step < self.display_height - self.hero_ball.r):
            y_change = -self.motion_step
        if action in [2,3,4] and (self.hero_ball.pos[0] + self.motion_step < self.display_width - self.hero_ball.r):
            x_change = self.motion_step
        if action in [8,7,6] and (self.hero_ball.pos[0] - self.motion_step > self.hero_ball.r):
            x_change = -self.motion_step

        # Move hero ball
        if action in [8,1,2]:
            self.hero_ball.update_position(0,self.motion_step)
        if action in [6,5,4]:
            self.hero_ball.update_position(0,-self.motion_step)
        if action in [8,7,6]:
            self.hero_ball.update_position(-self.motion_step,0)
        if action in [2,3,4]:
            self.hero_ball.update_position(self.motion_step,0)

        # Move balls to next position
        for ball in self.balls:
            if ball.live:
                ball.move(self.dt)

        reward = 0.0 # penalty incured at each time step
        # Calculate reward
        for ball in self.balls:
            if ball.live:
                # Check if Hero ball is hit
                if Ball.check_hit(self.hero_ball.pos, self.hero_ball.r, ball.pos, ball.r):
                    if ball.color == GREEN:
                        reward += 0.1 # Reward for hitting Green Ball
                        ball.live = False
                        self.ball_inventory['green'] -= 1
                    else:
                        reward = -1.0 # Penalty for hitting Red and Exit
                        ball.live = False
                        self.done = True
                        break

                # Check if any green or blue balls left
                if self.ball_inventory['green'] == 0: # Exit if no green balls left reached
                    self.done = True
                    break

        return self._get_state(), reward, self.done

    def render(self, gamedisplay=None):
        if gamedisplay:
            self.hero_ball.draw(gamedisplay)
            for ball in self.balls:
                if ball.live:
                    # Draw the ball
                    ball.draw(gamedisplay)
