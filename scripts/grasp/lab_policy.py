"""The grasp task's oracle, registered for --policy oracle (lab.tasks.TASKS, lab.policy).

It knows from the layout bank which face-square grip can be planned and aims
it at perception's estimate: what a policy could at best learn, since it reads
the one thing the policy cannot see. On a second attempt it turns to the other
face if both work.
"""

import numpy as np
import torch

from . import task as T


def oracle_actions(env):
    """(num_envs, ACT_DIM): the face the bank says works, aimed at the estimate."""
    u = env.unwrapped
    usable = u.bank.usable(u.task)
    acts = []
    for e in range(u.num_envs):
        feas = usable[u.layout[e]]
        face = 0 if feas[0] else 1
        if u.attempt[e] > 0 and feas.all():
            face = int(u.attempt[e] % 2 == 1) ^ face
        acts.append(T.oracle_action(u.est[e].cube[2], face))
    return torch.tensor(np.array(acts), dtype=torch.float32, device=u.device)
