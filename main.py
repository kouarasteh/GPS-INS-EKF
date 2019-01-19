#IMPORTANT READ ME!: Please run this in python3! If you run this in python2 you will get errors with the division.
#Python2 will be retired on Jan 1 2020 anyways... So if you haven't its time to move on.

import scipy.io as sio #used to import .mat
import numpy as np #used for general matrix computation
import matplotlib.pyplot as plt #used for plotting
from numpy import cos, sin
import time as clock #Used to benchmark how long it takes for the code to run
from quat2euler import quat2euler

# 6 DOF EKF GPS-INS Fusion
# This code was developed by Girish Chowdhary
# This code was then turned into python by Justin Wasserman
# The EKF implementation portion of this code was developed by Kourosh Arasteh
# To learn more about the filter, please read:
# 1. A Compact Guidance, Navigation, and Control System for Unmanned Aerial Vehicles (2006)
# by Henrik B. Christophersen , R. Wayne Pickell , James C. Neidhoefer , Adrian A. Koller , K. Kannan , Eric N. Johnson
# 2. (More advanced) GPS-Denied Indoor and Outdoor Monocular Vision Aided
# Navigation and Control of Unmanned Aircraft, Chowdhary, Magree, Johnson,
# Shein, Wu
# 3. (very detailed) (late) Nimrod Rooz's thesis proposal


# load a check file with the data
A = sio.loadmat('check.mat')
A = A['A']
# This loads a matrix called A (arbitrary name, nothing to do with the real
# A) which contains the data that you need

# initial position in x y and z
x=np.matrix([0, 0, 0])
x = np.ravel(x)

# bias values, these are accelerometer and gyroscope biases
pi = np.pi
bp= 0#.54*pi/180
bq=-12*pi/180
br=-.1*pi/180
bfx = 0
bfy = 0
bfz = 0


# IMU location specifier
r_imu=np.matrix([-.5/12, -3/12, 1/12]).T*0 ## I have set this to zero, for Bonus, you can include the effect of this
r_GPS=np.matrix([0, 0 ,0 ]) # This is the location of the GPS wrt CG, this is very important
#rotation matrix ------------------------------------------------------

tf= A.shape[1]
phi_raw= np.zeros((tf,))
theta_raw= np.zeros((tf,))
psi_raw =  np.zeros((tf,))

phi= np.zeros((tf,))
theta= np.zeros((tf,))
psi =  np.zeros((tf,))

#roation matrix body to inertial
L_bi = np.zeros((3,3))
L_bi[0,0] = cos(theta[0])*cos(psi[0])
L_bi[0,1] = cos(theta[0])*sin(psi[0])
L_bi[0,2] = -sin(theta[0])
L_bi[1,0] = sin(phi[0])*sin(theta[0])*cos(psi[0])-cos(phi[0])*sin(psi[0])
L_bi[1,1] = sin(phi[0])*sin(theta[0])*sin(psi[0])+cos(phi[0])*cos(psi[0])
L_bi[1,2] = sin(phi[0])*cos(theta[0])
L_bi[2,0] = cos(phi[0])*sin(theta[0])*cos(psi[0])+sin(phi[0])*sin(psi[0])
L_bi[2,1] = cos(phi[0])*sin(theta[0])*sin(psi[0])-sin(phi[0])*cos(psi[0])
L_bi[2,2] = cos(phi[0])*cos(theta[0])


Rt2b=L_bi
b = np.zeros((4,1))
[U,S,V]=np.linalg.svd(Rt2b)
R = U*V.T
if 1+R[0,0]+R[1,1]+R[2,2] > 0:
    b[0,0]    = 0.5*np.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    b[1,0]    = (R[2,1]-R[1,2])/4/b[0]
    b[2,0]    = (R[0,2]-R[2,0])/4/b[0]
    b[3,0]    = (R[1,0]-R[0,1])/4/b[0]
    b       = b/np.linalg.norm(b)    # renormalize
else:
    print(R)
    print('R diagonal too negative.')


#b =[0 0 0 0]'

# set quats
#-----------------------------------------------------------------
q1=b[0]#the quaternions are called b1-b4 in the data file that you loaded
q2=b[1]
q3=b[2]
q4=b[3]

#initialize velocity
vx = 0
vy = 0
vz = 0

#set sample time
dt = .01

xhatR = np.zeros((tf, 16))
P_R = np.zeros((tf, 16))
z_R = np.zeros((tf, 6))
OMEGA_raw = np.zeros((tf,3))
OMEGA = np.zeros((tf,3))
FX = np.zeros((tf,3))


# initialize x hat
# Note carefull the order the states appear in, this can be arbitrary, but
# we must stick to it along the entire code
#      [x y z vx vy vz          quat    gyro-bias accl-bias]
xhat = np.matrix([0, 0, 0, 0, 0, 0, b[0], b[1], b[2], b[3], bp, bq, br, bfx, bfy, bfz])
xhat = np.ravel(xhat)

# noise params process noise (my gift to you :))
Q = np.diag([.1, .1, .1, .1, .1, .1, .8, .8, .8, .8, .0001, .0001, .0001, .0001, .0001, .0001])

# noise params, measurement noise
# measurements are GPS position and velocity and mag
#R = np.diag([9, 9, 9, 3, 3, 3])
R = np.diag([10,10,10,8,8,8])
# Initialize P, the covariance matrix
P = np.diag([30, 30, 30, 3, 3, 3, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
Pdot=P*0
start_time = clock.time()
for k in range(0,tf):
    time= (k+1)*dt

    #  Streaming sensor measurements and adjust for bias
    # these come from the file that is loaded in the begining
    p = (A[0,k]*(pi/180)-xhat[10])
    q = (A[1,k]*(pi/180)-xhat[11])
    r = A[2,k]*(pi/180)-xhat[12]
    fx = (A[3,k]-xhat[13])
    fy = (A[4,k]-xhat[14])
    fz = -A[5,k]-xhat[15] + 32.2
    # Raw sensor measurments for plotting
    p_raw = A[0,k]*(pi/180)
    q_raw = A[1,k]*(pi/180)
    r_raw = A[2,k]*(pi/180)
    fx_raw = A[3,k]
    fy_raw = A[4,k]
    fz_raw = A[5,k]

    quat = np.array([xhat[6], xhat[7], xhat[8], xhat[9]])

    q1 = quat[0]
    q2 = quat[1]
    q3 = quat[2]
    q4 = quat[3]
    L_lb = np.zeros((3,3))
    L_lb[0,0] = q1**2+q2**2-q3**2-q4**2
    L_lb[0,1] = 2*(q2*q3+q1*q4)
    L_lb[0,2] = 2*(q2*q4-q1*q3)
    L_lb[1,0] = 2*(q2*q3-q1*q4)
    L_lb[1,1] = q1**2-q2**2+q3**2-q4**2
    L_lb[1,2] = 2*(q3*q4+q1*q2)
    L_lb[2,0] = 2*(q2*q4+q1*q3)
    L_lb[2,1]  = 2*(q3*q4-q1*q2)
    L_lb[2,2] = q1**2-q2**2-q3**2+q4**2
    L_bl = L_lb.T
    om = np.array([[0,p,q,r],
          [-p,0,-r,q],
          [-q,r,0,-p],
          [-r,-q,p,0]])
    ## Implement your code here:
    ## Prediction step
    #First write out all the dots, e.g. pxdot, pydot, q1dot etc
    pxdot = (xhat[0] + xhat[3])
    pydot = (xhat[1] + xhat[4])
    pzdot = (xhat[2] + xhat[5])
    vxdot,vydot,vzdot = np.sum(L_bl*[fx,fy,fz],axis=1)
    [q1dot,q2dot,q3dot,q4dot] = -0.5 * np.matmul(om,[q1,q2,q3,q4])
    bpdot = 0
    bqdot = 0
    brdot = 0
    bxdot = 0
    bydot = 0
    bzdot = 0
    xdot = np.array([pxdot,pydot,pzdot,vxdot,vydot,vzdot,q1dot,q2dot,q3dot,q4dot,bpdot,bqdot,brdot,bxdot,bydot,bzdot])
    #Now integrate Euler Integration for Process Updates and Covariance Updates
    # Euler works fine
    xhat = xhat + xdot*dt

    # Extract and normalize the quat
    quat = np.matrix([xhat[6], xhat[7], xhat[8], xhat[9]])

    quatmag= np.linalg.norm(quat)#sqrt(q1^2+q2^2+q3^2+q4^2)
    #Renormalize quaternion if needed
    if abs(quatmag-1)>0.01:
        quat = quat/quatmag

    #re-assign quat
    xhat[6] = quat[0,0]
    xhat[7] = quat[0,1]
    xhat[8] = quat[0,2]
    xhat[9] = quat[0,3]
    q1 = xhat[6]
    q2 = xhat[7]
    q3 = xhat[8]
    q4 = xhat[9]
    #Remember again the state vector [ px py pz vx vy vz q1 q2 q3 q4 bp bq br bx by bz]

    # Now write out all the partials to compute the transition matrix Phi
    #delV/delQ
    dvdq = 2*np.array([[(q1*fx - q4*fy + q3*fz),(q2*fx + q3*fy + q4*fz),(-q3*fx + q2*fy + q1*fz),(-q4*fx - q1*fy + q2*fz)],
            [(q4*fx + q1*fy - q2*fz),(q3*fx - q2*fy - q1*fz),(q2*fx + q3*fy + q4*fz),(q1*fx - q4*fy + q3*fz)],
            [(-q3*fx + q2*fy + q1*fz),(q4*fx + q1*fy - q2*fz),(-q1*fx + q4*fy - q3*fz),(q2*fx + q3*fy + q4*fz)]])

    #delV/del_abias
    dvdba = -1 * L_bl
    #delQ/del_gyrobias
    dqdbw = -0.5 * np.array([[q2,q3,q4],
                    [-q1,q4,-q3],
                    [-q4,-q1,q2],
                    [q3,-q2,-q1]])
    #delV/del_gyro_bias
    dvdbw = np.matmul(dvdq,dqdbw)
    #delQ/delQ
    dqdq = -0.5 * om

    # Now assemble the Transition matrix
    z33 = np.zeros((3,3))
    z43 = np.zeros((4,3))
    z34 = np.zeros((3,4))
    z36 = np.zeros((3,6))
    i33 = np.identity(3)
    Phi = np.concatenate((np.concatenate((z33,i33,z34,z33,z33),axis=1),
                      np.concatenate((z33,z33,dvdq,z33,dvdba),axis=1),
                      np.concatenate((z43,z43,dqdq,dqdbw,z43),axis=1),
                      np.concatenate((z33,z33,z34,z33,z33),axis=1),
                      np.concatenate((z33,z33,z34,z33,z33),axis=1)),axis=0)

   # Propagate the error covariance matrix, I suggest using the continuous integration since Q, R are not discretized
    elem1 = np.matmul(Phi,P)
    elem2 = np.matmul(elem1,Phi.T)
    elem3 = elem2 + Q
    P = elem3
    ## Correction step
    # Get your measurements, 3 positions and 3 velocities from GPS
    z =np.matrix([ A[6,k], A[7,k], A[8,k], A[9,k], A[10,k], A[11,k]]) # x y z vx vy vz
    z = np.ravel(z)
    # Write out the measurement matrix linearization to get H
    r1 = r_GPS.item(0)
    Hxq = np.array([[-q1,-q2,q3,q4],
                      [-q4,-q3,-q2,-q1],
                      [q3,-q4,q1,-q2]])
    Hxq = Hxq * 2 * r1
    Hvq = np.array([[ q3*q + q4*r, q4*q - q3*r, q1*q-q2*r,q2*q + q1*r],
           [-q2*q - q1*r, q2*r - q1*q, q4*q-q3*r,q3*q + q4*r],
           [ q1*q - q2*r,-q2*q - q1*r,-q3*q-q4*r,q4*q - q3*r]])
    Hvq = Hvq * 2 * r1
    #del P/del q

    # del v/del q

    # Assemble H
    H = np.concatenate((np.concatenate((i33,z33,Hxq,z36),axis=1),
                      np.concatenate((z33,i33,Hvq,z36),axis=1)),axis=0)

    #Compute Kalman gain
    elem1 = np.matmul(P,H.T)
    elem2 = np.matmul(H,P)
    elem3 = np.matmul(elem2,H.T)
    Sk = (elem3 + R)
    elem4 = np.linalg.inv(Sk)
    Lk = np.matmul(elem1,elem4)
    # Perform xhat correction    xhat = xhat + K*(z - H*xhat)
    elem5 = (z - np.matmul(H,xhat))
    xhat = xhat + np.matmul(Lk,elem5)
    # propagate error covariance approximation P = (eye(16)-K*H)*P
    #  end
    elem6 = np.matmul(Lk,Sk)
    elem7 = np.matmul(elem6,Lk.T)
    P = P - elem7

    ## Now let us do some book-keeping
    # Get some Euler angles
    phi_,theta_,psi_ =quat2euler(quat)
    phi[k] = phi_
    theta[k] = theta_
    psi[k] = psi_
    phi[k]=phi[k]*(180/pi)
    theta[k]=theta[k]*(180/pi)
    psi[k]=psi[k]*(180/pi)

    quat1 = A[12:16,k]
    quat1 = np.reshape(quat1, (1,4))
    phi_raw_,theta_raw_,psi_raw_=quat2euler(quat1)
    phi_raw[k] = (phi_raw_)
    theta_raw[k] = (theta_raw_)
    psi_raw[k] = (psi_raw_)
    phi_raw[k]=phi_raw[k]*(180/pi)
    theta_raw[k]=theta_raw[k]*(180/pi)
    psi_raw[k]=psi_raw[k]*(180/pi)

    #  Recording data for plots
    xhatR[k,:]= xhat
    P_R[k,:] = np.diag(P)
    z_R[k,:] = z.T
    OMEGA_raw[k,:]=(np.matrix([p_raw,q_raw,r_raw]).flatten())
    OMEGA[k,:]= np.matrix([p,q,r]).flatten()
    FX[k,:]=np.matrix([fx_raw,fy_raw,fz_raw]).flatten()

print("Took {} seconds".format(clock.time() - start_time))
t = range(0,tf)
plt.figure(1)
plt.plot(t,P_R[:,0], label = 'px')
plt.plot(t,P_R[:,1], label = 'py')
plt.plot(t,P_R[:,2], label = 'pz')
plt.title('Covariance of Position')
plt.legend()

plt.figure(2)
plt.plot(t,P_R[:,3], label = 'pxdot')
plt.plot(t,P_R[:,4], label = 'pydot')
plt.plot(t,P_R[:,5], label = 'pzdot')
plt.legend()
plt.title('Covariance of Velocities')

plt.figure(3)
plt.plot(t,P_R[:,6])
plt.plot(t,P_R[:,7])
plt.plot(t,P_R[:,8])
plt.plot(t,P_R[:,9])
plt.title('Covariance of Quaternions')
#8
plt.figure(8)
plt.plot(t, phi,  label = 'phi')
plt.plot(t, theta, label = 'theta')
plt.plot(t, psi,  label = 'psi')
plt.plot(t, phi_raw,'b:', label = 'phiraw')
plt.plot(t, theta_raw,'g:', label = 'thetaraw')
plt.plot(t, psi_raw,'r:', label = 'psiraw')
plt.title('Phi, Theta, Psi')
plt.legend()
#4
plt.figure(4)
#plt.plot(t,xhat[:,0:2], t, A[6:8,:])
plt.plot(t,xhatR[:,0])
plt.plot(t,xhatR[:,1])
plt.plot(t,xhatR[:,2])
plt.plot(t,A[6,:], 'r:')
plt.plot(t,A[7,:], 'r:')
plt.plot(t,A[8,:], 'r:')
#plot(t,z_R(:,1),'r')
plt.title('Position')
#5
plt.figure(5)
plt.plot(t,xhatR[:,3])
plt.plot(t,xhatR[:,4])
plt.plot(t,xhatR[:,5])
plt.plot(t,A[9,:], 'r:')
plt.plot(t,A[10,:], 'r:')
plt.plot(t,A[11,:], 'r:')
plt.title('vel x y z')

plt.figure(6)
plt.plot(t,xhatR[:,6])
plt.plot(t,xhatR[:,7])
plt.plot(t,xhatR[:,8])
plt.plot(t,xhatR[:,9])
plt.plot(t, A[12,:], 'r:')
plt.plot(t, A[13,:], 'r:')
plt.plot(t, A[14,:], 'r:')
plt.plot(t, A[15,:], 'r:')
plt.title('Quat')

plt.figure(9)
plt.plot(t,xhatR[:,10:15])
plt.title('Bias')
plt.legend(['bp','bq','br','bfx','bfy','bfz'])

plt.figure(7)
plt.plot(t,OMEGA[:,0],t,OMEGA[:,1],t,OMEGA[:,2])
plt.title('OMEGA with Bias')
plt.legend(['p','q','r'])

plt.figure(10)
plt.plot(t,OMEGA_raw[:,0],t,OMEGA_raw[:,1],t,OMEGA_raw[:,2])
plt.title('OMEGA raw without Bias')
plt.legend(['p','q','r'])

plt.figure(11)
plt.plot(t,FX[:,0],t,FX[:,1],t,FX[:,2])
plt.title('accelerometer')
plt.legend(['ax','ay','az'])
plt.show()
