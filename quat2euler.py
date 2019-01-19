import numpy as np

def quat2euler(q): #q is a matrix of size n x 4 where n is the number of quaternions
    n = q.shape[0]
    
    m = np.eye(3)
    phi = np.zeros((n,1))
    theta= np.zeros((n,1))
    psi = np.zeros((n,1))

    for i in range(n):
        q0 = q[i,0]
        q1 = q[i,1]
        q2 = q[i,2]
        q3 = q[i,3]
        m[0,0] = 1.0 - 2.0*( q2*q2 + q3*q3 )
        m[0,1] = 2.0*( q1*q2 - q0*q3 )
        m[0,2] = 2.0*( q1*q3 + q0*q2 )
        m[1,0] = 2.0*( q1*q2 + q0*q3 )
        m[1,1] = 1.0 - 2.0*( q1*q1 + q3*q3 )
        m[1,2] = 2.0*( q2*q3 - q0*q1 )
        m[2,0] = 2.0*( q1*q3 - q0*q2 )
        m[2,1] = 2.0*( q2*q3 + q0*q1 )
        m[2,2] = 1.0 - 2.0*( q1*q1 + q2*q2 )
        phi[i,0]   = np.arctan2( m[2,1], m[2,2] )
        theta[i,0] = -np.arcsin( m[2,0] )
        psi[i,0]   = np.arctan2( m[1,0], m[0,0] )
    return phi, theta, psi

if __name__ == "__main__":
    test = np.matrix([[0.756, 0.378, 0.378, 0.378], [0.857, 0.421, 0.253, 0.156], [1.0, 0.0, 0.0, 0.0]])
    phi, theta, psi = quat2euler(test)
    results = np.concatenate((phi, theta, psi), axis = 1)
    print(results)

    
        