from envs.car_racing import CarRacing

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
    
    def reset_car(self, x, y):
        state = super().reset()
        self.car.hull.position = (x, y) # Overwrite the car position
        return state

    def close(self):
        super().close()
