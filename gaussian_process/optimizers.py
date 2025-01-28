import numpy as np

def search(x_init, f_init, e_init, energy_force, learning_rate, beta=0.5, gamma=0.1):
    grad_dot = f_init@f_init  # Gradient squared
    t = learning_rate
    n_s = 0
    # Backtracking step size adjustment
    while energy_force(x_init + t * f_init)[0] > e_init - gamma * t * grad_dot:
        t *= beta  # Reduce step size
        n_s += 1
    return t, n_s

def linesearch_simple(guess, energy_force, prec, nsteps = 2000):
    x0 = guess.copy()
    pot0, f0 = energy_force(x0)
    n_search = 0
    for i in range(nsteps):
        step, n_add = search(x0, f0, pot0, energy_force, learning_rate = 0.1)
        n_search += n_add
        x1 = x0 + step * f0
        pot1, f1 = energy_force(x1)
        x0 = x1.copy()
        f0 = f1.copy()
        pot0 = pot1
    
    if np.linalg.norm(f1 - f0) < prec:
            return x1, i + n_search
    
    return x1, -1, i

def conj_gradient(guess, energy_force, prec = 1e-4, nsteps = 200, path = None):
    x1 = guess.copy()
    pot1, f1 = energy_force(x1)
    h1 = f1.copy()
    n_eval = 1
    for conj_i in range(nsteps):
        for conjugation in range(len(guess)): #dimensionality
            x0 = x1.copy()
            f0 = f1.copy()
            h0 = h1.copy()
            pot0 = pot1
            step, n_add = search(x0, h0, pot0, energy_force, learning_rate = 0.1)
            
            x1 = x0 + step * h0
            pot1, f1 = energy_force(x1)
            n_eval += n_add + 1
            if path is not None:
                path.x_steps.append(x1)
                path.nloglike.append(pot1)

            if np.linalg.norm(f1 - f0) < prec:
                return x1, pot1, n_eval
            
            gamma = (f1 - f0)@f1/(f0@f0)
            h1 = f1 + gamma*h0
        h1 = f1.copy() #restarts conjugation
        
    #Didnt converge
    return x1, pot1, -1

def kinetic(p, mass):
        if hasattr(p, '__len__'):
            return p@p/(2*mass)
        else:
            return p**2/(2*mass)

def velo_step(x, p, f, energy_force, mass, dt):
    p_half = p + dt/2 * f
    x_new = x + dt/mass * p_half
    pot_new, f_new = energy_force(x_new)
    p_new = p_half + dt/2 * f_new
    
    return x_new, p_new, pot_new, f_new

def velo_integrate(x0, p0, f0, energy_force, mass, dt, steps):
    x1 = x0.copy()
    p1 = p0.copy()
    f1 = f0.copy()
    
    for velo in range(steps):
        x1, p1, pot1, f1 = velo_step(x1, p1, f1, energy_force, mass, dt)
        
    return x1, p1, pot1, f1

def HMC(guess, energy_force, nsteps = 20, n_integrate = 10, mass = 1, sigma = 1, dt = 1e-2, path = None):
    dim = len(guess)
    x0 = np.array(guess).copy()
    p0 = np.random.normal(0.0, sigma, size = dim)
    pot0, f0 = energy_force(x0)
    for momentum_step in range(nsteps):
        x1, p1, pot1, f1 = velo_integrate(x0, p0, f0, energy_force, mass, dt, n_integrate)
        prob = 1.0 if kinetic(p0, mass) + pot0 - (kinetic(p1, mass) + pot1) > 0 else np.exp(kinetic(p0, mass) + pot0 - (kinetic(p1, mass) + pot1))
        
        p0 = np.random.normal(0.0, sigma, size = dim)
        if np.random.random() < prob:
            if path is not None:
                path.x_steps.append(x0)
                path.nloglike.append(pot0)
                ratio = (momentum_step + 1)/nsteps
                print(str(int(100*ratio)) + '%', end = ' ', flush = True)
            x0 = x1.copy()
            f0 = f1.copy()
            pot0 = pot1
        
    return x0, pot0, n_integrate * nsteps