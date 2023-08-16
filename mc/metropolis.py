import torch
import torch.nn.functional as F
import numpy as np


class Chain:
    def __init__(self, action, x0, delta=1., temperature=1.):
        self.action = action
        self.x = x0
        self.S = self.action(self.x)
        self.delta = delta
        self.temperature = temperature
        self._recent = [False]

    def _propose(self, x, delta):
        xp = x + delta * torch.randn(x.size())

        return xp

    def _acceptreject(self, temperature, x, S, xp, Sp):
        Sdiff = Sp - S
        acc = torch.rand(1) < torch.exp(-Sdiff / temperature)

        x = torch.where(acc, xp, x)
        S = torch.where(acc, Sp, S)
        accepted = torch.where(acc, torch.tensor(True), torch.tensor(False))

        return x, S, accepted

    def _action(self, x, delta):
        xp = self._propose(x, delta)
        Sp = self.action(xp).real
        return xp, Sp

    def step(self, N=1):
        self.S = self.action(self.x).real
        for _ in range(N):
            xp, Sp = self._action(self.x, self.delta)
            self.x, self.S, accepted = self._acceptreject(
                self.temperature, self.x, self.S, xp, Sp)
            self._recent.append(accepted)
        self._recent = self._recent[-100:]

    def calibrate(self):
        # Adjust delta.
        self.step(N=100)
        while self.acceptance_rate() < 0.3 or self.acceptance_rate() > 0.55:
            if self.acceptance_rate() < 0.3:
                self.delta *= 0.98
            if self.acceptance_rate() > 0.55:
                self.delta *= 1.02
            self.step(N=100)

    def acceptance_rate(self):
        return sum(self._recent) / len(self._recent)

    def iter(self, skip=1):
        while True:
            self.step(N=skip)
            yield self.x
