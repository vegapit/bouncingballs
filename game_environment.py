import numpy, gymnasium
from ball import Ball
from player_ball import PlayerBall
from typing import Optional

RED = (255,0,0)
GREEN = (0,102,0)
BLUE = (0,128,255)
COLORS = {'red': RED, 'green': GREEN}

def generate_random_speed_vector(minspeed, maxspeed):
    vx, vy = numpy.random.randint(minspeed, maxspeed+1), numpy.random.randint(minspeed, maxspeed+1)
    if numpy.random.random_sample() < 0.5:
        vx = -vx
    if numpy.random.random_sample() < 0.5:
        vy = -vy
    return [vx,vy]

def generate_random_starting_position(maxdistance, display_height, display_width):
    x, y = numpy.random.randint(8, maxdistance+1), numpy.random.randint(8, maxdistance+1)
    if numpy.random.random_sample() < 0.5:
        x = display_width - x
    if numpy.random.random_sample() < 0.5:
        y = display_height - y
    return [x,y]

class GameEnvironment( gymnasium.Env ):

    # --- ADDED METADATA FOR RENDER MODE SUPPORT ---
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, display_shape, dt):
        self.display_width, self.display_height = display_shape[0], display_shape[1]
        self.dt = dt
        self.motion_step = 5
        self.max_speed = 150

        self.action_space = gymnasium.spaces.Discrete(9) # 0: No move, 1: Up, 2: Up&Right, 3: Right, 4: Down&Right, 5: Down, 6: Down&Left, 7: Left, 8: Up&Left

        self.observation_space = gymnasium.spaces.Dict({
            'hero': gymnasium.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=numpy.float32),
            'balls_position': gymnasium.spaces.Box(low=0.0, high=1.0, shape=(10,2), dtype=numpy.float32),
            'balls_speed': gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(10,2), dtype=numpy.float32),
            'balls_color': gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype=numpy.int32),
            'balls_status': gymnasium.spaces.Box(low=0, high=1, shape=(10,), dtype=numpy.int32),
        })

        self.balls = []
        self.hero_ball = None
        self.ball_inventory = {'red': 4, 'green': 6}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed = seed)

        self.balls = []
        self.step_counter = 0

        # NOTE: Using a fixed seed for demonstration, but typically let super().reset() handle it
        if seed is not None:
            numpy.random.seed(seed)

        #Balls
        self.ball_inventory = {'red': 4, 'green': 6}
        for key, count in self.ball_inventory.items():
            for i in range( count ):
                init_pos = generate_random_starting_position(20, self.display_height, self.display_width)
                init_speed = generate_random_speed_vector(self.max_speed / 2, self.max_speed)
                ball = Ball(init_pos, init_speed, COLORS[key], (self.display_width, self.display_height))
                self.balls.append( ball )

        #Players balls
        init_hero_pos = [self.display_width/2,self.display_height/4]
        self.hero_ball = PlayerBall(init_hero_pos,BLUE,(self.display_width,self.display_height))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        dim_array = numpy.array([self.display_width, self.display_height], dtype=numpy.float32)

        hero_position_obs = self.hero_ball.pos / dim_array
        
        balls_position_obs = numpy.zeros( (len(self.balls),2), dtype=numpy.float32 )
        balls_speed_obs = numpy.zeros( (len(self.balls),2), dtype=numpy.float32 )
        balls_color_obs = numpy.zeros( len(self.balls), dtype=numpy.int32 )
        balls_status_obs = numpy.zeros( len(self.balls), dtype=numpy.int32 )
        
        for i, ball in enumerate(self.balls):
            balls_position_obs[i,:] = ball.pos / dim_array
            balls_speed_obs[i,:] = ball.v / float( self.max_speed )
            balls_color_obs[i] = int(ball.color == GREEN)
            balls_status_obs[i] = int(ball.live)
                
        return {
            'hero': hero_position_obs,
            'balls_position': balls_position_obs,
            'balls_speed': balls_speed_obs,
            'balls_color': balls_color_obs,
            'balls_status': balls_status_obs
        }

    def _get_info(self):
        return self.ball_inventory

    def step(self, action):
        terminated, truncated = False, False
        x_change, y_change, reward = 0.0, 0.0, 0.0

        self.step_counter += 1
        if self.step_counter >= 1000:
            truncated = True

        # 0: No move, 1: Up, 2: Up&Right, 3: Right, 4: Down&Right, 5: Down, 6: Down&Left, 7: Left, 8: Up&Left
        match action:
            case 1:  # Up
                x_change, y_change = 0, -self.motion_step
            case 2:  # Up&Right
                x_change, y_change = self.motion_step, -self.motion_step
            case 3:  # Right
                x_change, y_change = self.motion_step, 0
            case 4:  # Down&Right
                x_change, y_change = self.motion_step, self.motion_step
            case 5:  # Down
                x_change, y_change = 0.0, self.motion_step
            case 6:  # Down&Left
                x_change, y_change = -self.motion_step, self.motion_step
            case 7:  # Left
                x_change, y_change = -self.motion_step, 0.0
            case 8:  # Up&Left
                x_change, y_change = -self.motion_step, -self.motion_step
            case _:
                pass

        max_distance = numpy.linalg.norm([self.display_width, self.display_height])

        # Update hero ball position (only once)
        self.hero_ball.update_position(x_change, y_change)

        # Move balls to next position
        for ball in self.balls:
            ball.update_position( self.dt )

        # Calculate reward
        for ball in self.balls:
            if ball.live:
                # Check if Hero ball is hit
                if Ball.check_hit(self.hero_ball.pos, self.hero_ball.r, ball.pos, ball.r):
                    if ball.color == GREEN:
                        reward += 0.25 # Reward for hitting Green Ball
                        ball.live = False
                        self.ball_inventory['green'] -= 1
                    else:
                        reward = -1.0 # Penalty for hitting Red and Exit
                        ball.live = False
                        terminated = True
                        break

                # Check if any green or blue balls left
                if self.ball_inventory['green'] == 0: # Exit if no green balls left reached
                    reward += 1.0
                    terminated = True
                    break

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, gamedisplay):
        if gamedisplay:
            self.hero_ball.draw(gamedisplay)
            for ball in self.balls:
                if ball.live:
                    # Draw the ball
                    ball.draw(gamedisplay)
