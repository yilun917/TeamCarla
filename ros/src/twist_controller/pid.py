
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        self.cur_scale_step = 0

        self.int_val = self.last_error = 0.0

    def reset(self):
        self.int_val = 0.0

    def step(self, error, sample_time):

        if self.last_error < 0.0001:
            self.last_error = error

        integral = self.int_val + error * sample_time
        derivative = (error - self.last_error) / sample_time

        val = self.kp * error + self.ki * integral + self.kd # * derivative

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val

    # kp = 0.001 #0.3  0.5 Need to tune
    # ki = 0.01 #0.1  0.1
    # kd = 0.0 #0.0  2.5
