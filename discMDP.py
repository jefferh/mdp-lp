
# LP Formulation of Discounted MDP

import pyomo.environ as pyo

model = pyo.AbstractModel()

# Sets
model.S = pyo.Set() # state set
model.GrA = pyo.Set(dimen=2) # set of all state-action pairs

def A_init(model, state):
    retval = []
    for (s,a) in model.GrA:
        if s == state:
            retval.append(a)
    return retval
model.A = pyo.Set(model.S, initialize=A_init) # action sets

# Parameters
model.c = pyo.Param(model.GrA) # one-step costs
model.p = pyo.Param(model.GrA, model.S, default=0) # transition probabilities
model.disc = pyo.Param(within=pyo.UnitInterval) # discount factor

# Variables
model.x = pyo.Var(model.GrA, domain=pyo.NonNegativeReals) # occupancy measure

# Objective
def OBJ_rule(model):
    return pyo.summation(model.c, model.x)
model.OBJ = pyo.Objective(rule=OBJ_rule, sense=pyo.minimize)

# Constraints
def CON_rule(model, state):
    return sum(model.x[state,a] for a in model.A[state]) \
    - model.disc*sum(model.p[sa,state]*model.x[sa] for sa in model.GrA) \
    == 1
model.CON = pyo.Constraint(model.S, rule=CON_rule)
