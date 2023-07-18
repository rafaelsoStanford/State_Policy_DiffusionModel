from envs.car_racing import CarRacing
from envs.car_dynamics import Car
import numpy as np
import Box2D

FPS = 50  # Frames per second

class EnvWrapper(CarRacing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def step_noRender(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        info = {
                'car_position_vector': self.return_carPosition(),
                'car_velocity_vector': self.car.hull.linearVelocity,
            }

        return info
    
    def reset_car(self, x, y, xdot, ydot, initAngle, omega, phase):
        state = super().reset()
        

        init_angle =  initAngle #np.arctan2(ydot, xdot) - np.pi / 2
        self.car = Car(self.world, init_angle=float(init_angle), init_x= float(x), init_y=float(y))
        self.car.hull.linearVelocity = Box2D.b2Vec2(float(xdot),float(ydot))
        v = np.sqrt(xdot**2 + ydot**2)
        
        for idx, w in enumerate(self.car.wheels):
            w.angularVelocity = v
            w.omega = v / (w.wheel_rad)

        
    def close(self):
        super().close()
