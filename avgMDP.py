
# LP Formulation of Average-Reward MDP

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
model.r = pyo.Param(model.GrA) # one-step rewards
model.p = pyo.Param(model.GrA, model.S, default=0) # transition probabilities
model.alpha = pyo.Param(model.S) # initial distribution

# Variables
model.x = pyo.Var(model.GrA, domain=pyo.NonNegativeReals) # occupancy measure
model.y = pyo.Var(model.GrA, domain=pyo.NonNegativeReals)

# Objective
def OBJ_rule(model):
    return pyo.summation(model.r, model.x)
model.OBJ = pyo.Objective(rule=OBJ_rule, sense=pyo.maximize)

# Constraints
def x_rule(model, state):
    return sum(model.x[state,a] for a in model.A[state]) \
    - sum(model.p[sa,state]*model.x[sa] for sa in model.GrA) \
    == 0
model.x_rule = pyo.Constraint(model.S, rule=x_rule)
def y_rule(model,state):
    return sum(model.x[state,a] for a in model.A[state]) \
    + sum(model.y[state,a] for a in model.A[state]) \
    - sum(model.p[sa,state]*model.y[sa] for sa in model.GrA) \
    == model.alpha[state]
model.y_rule = pyo.Constraint(model.S, rule=y_rule)
