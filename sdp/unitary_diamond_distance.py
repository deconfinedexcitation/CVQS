def diamond(uu,vv,d):
    #uu and vv are d \times d unitaries.
    # Define and solve the CVXPY problem.
    # Create a complex matrix variable.
    Y = cp.Variable((2,2),complex=True)
    rho0=cp.Variable((d,d),complex=True)
    rho1=cp.Variable((d,d),complex=True)
    psi0=cp.bmat([[cp.trace(rho0),cp.trace(uu@rho0@(vv.conj().T))],[cp.trace(vv@rho0@(uu.conj().T)),cp.trace(rho0)]])
    psi1=cp.bmat([[cp.trace(rho1),-cp.trace(uu@rho1@(vv.conj().T))],[-cp.trace(vv@rho1@(uu.conj().T)),cp.trace(rho1)]])
    rr=cp.bmat([[psi0,Y],[cp.conj(cp.transpose(Y)),psi1]])
    constraints = [rr >> 0]
    constraints += [cp.trace(rho0) == 1]
    constraints += [cp.trace(rho1) == 1]
    constraints += [rho0 >> 0]
    constraints += [rho1 >> 0]


    prob = cp.Problem(cp.Maximize( (1/2)*( cp.real(  cp.trace(Y)+cp.trace(cp.conj(cp.transpose(Y)))  ) )  ),constraints)

    prob.solve(solver=cp.SCS,verbose=True)
    return [Y.value,rho0.value,rho1.value,prob.value]
