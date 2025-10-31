import numpy

RED = (255,0,0)
GREEN = (0,102,0)

class ExpertAgent:
    def __init__(self, x_step, y_step, dt):
        self.x_step = x_step
        self.y_step = y_step
        self.dt = dt

    def select_action(self, hero_ball, balls):
        scores = numpy.zeros(9, dtype=numpy.float32)  # 0-8 actions
        next_balls_pos = [ ball.next_position( self.dt ) for ball in balls ]

        for k in range(9):
            x_change, y_change = self._get_steps_from_action(k)
            nxt_hero_pos = hero_ball.next_position(x_change, y_change)
            for (nxt_pos, ball) in zip(next_balls_pos, balls):
                if ball.live:
                    if ball.color == GREEN:
                        value = 0.25
                    else:
                        value = -1.0
                    scores[k] += self._score(nxt_hero_pos, nxt_pos, value)
        
        return int( numpy.argmax( scores ) )

    def _score(self, hero_pos, ball_pos, value):
        sq_dist = numpy.linalg.norm(hero_pos - ball_pos)
        return value * numpy.exp(-sq_dist)

    def _get_steps_from_action(self, action):
        # 0: No move, 1: Up, 2: Up&Right, 3: Right, 4: Down&Right, 5: Down, 6: Down&Left, 7: Left, 8: Up&Left
        match action:
            case 1:  # Up
                return 0, -self.y_step
            case 2:  # Up&Right
                return self.x_step, -self.y_step
            case 3:  # Right
                return self.x_step, 0
            case 4:  # Down&Right
                return self.x_step, self.y_step
            case 5:  # Down
                return 0.0, self.y_step
            case 6:  # Down&Left
                return -self.x_step, self.y_step
            case 7:  # Left
                return -self.x_step, 0.0
            case 8:  # Up&Left
                return -self.x_step, -self.y_step
            case _:
                return 0.0, 0.0
