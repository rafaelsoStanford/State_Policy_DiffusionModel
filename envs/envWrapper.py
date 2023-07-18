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
    
    def reset_car(self, x, y, xdot, ydot, initAngle):
        # state = super().reset()
        

        # init_angle =  initAngle #np.arctan2(ydot, xdot) - np.pi / 2
        # self.car = Car(self.world, init_angle=float(init_angle), init_x= float(x), init_y=float(y))
        # self.car.hull.linearVelocity = Box2D.b2Vec2(float(xdot),float(ydot))
        # v = float(np.sqrt(xdot**2 + ydot**2))
        
        # for idx, w in enumerate(self.car.wheels):
        #     w.angularVelocity = v
        #     w.omega = v / (w.wheel_rad)

                # Delete all points in lists
        self.t1.clear()
        self.t2.clear()
        self.t3.clear()
        self.t4.clear()
        self.t5.clear()
        
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        

        init_angle = initAngle
        self.car = Car(self.world, init_angle=float(init_angle), init_x= float(x), init_y=float(y))

        self.car.hull.linearVelocity = Box2D.b2Vec2(float(xdot),float(ydot))
        v = float(np.sqrt(float(xdot)**2 + float(ydot)**2))    
        for w in self.car.wheels:
            w.angularVelocity = v
            w.omega = v / (w.wheel_rad)
        return self.step(None)[0]

        
    def close(self):
        super().close()
