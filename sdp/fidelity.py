def fidel(state1,state2,d):
    # state1 and state2 are d \times q quantum states
    # Define and solve the CVXPY problem.
    # Create a complex matrix variable.
    X = cp.Variable((d,d),complex=True)
    # The operator >> denotes matrix inequality.
    rr=cp.bmat([[state1,X],[cp.conj(cp.transpose(X)),state2]])
    constraints = [rr >> 0]


    prob = cp.Problem(cp.Maximize( (1/2)*( cp.real(  cp.trace(X)+cp.trace(cp.conj(cp.transpose(X)))  ) )  ),constraints)

    prob.solve(solver=cp.SCS,verbose=True)
    return [X.value,prob.value]
