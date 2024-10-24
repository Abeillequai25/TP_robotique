# -*- coding: utf-8 -*-
"""
Way Point navigtion

(c) S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr


# robot
x0 = 0.0
y0 = 0.0
theta0 = 0.0
robot = rob.Robot(x0, y0, theta0)


# position control loop timer
positionCtrlPeriod = 0.2
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# orientation control loop timer
orientationCtrlPeriod = 0.05
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)


# list of way points: list of [x coord, y coord]
WPlist = [ [2.0,2.0] ,[2.0, -2.0] , [-2.0, -2.0] , [-2.0, 2.0] , [2.0, 2.0] , [0.0, 0.0]]  
#WPlist = [ [2.0,2.0] , [2.0, -2.0] ]
#threshold for change to next WP
epsilonWP = 1
# init WPManager
WPManager = rob.WPManager(WPlist, epsilonWP)


# duration of scenario and time step for numerical integration
t0 = 0.0
tf = 25.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)


# initialize control inputs
Vr = 0.0
thetar = 0.0
omegar = 0.0
k1 = 1.0  # Position control gain
k2 = 20.0  # Orientation control gain

# loop on simulation time
for t in simu.t: 
   
    # WP navigation: switching condition to next WP of the list
    if WPManager.distanceToCurrentWP(robot.x, robot.y) < epsilonWP:
       WPManager.switchToNextWP()
        # !!!!! 
        # A COMPLETER EN TD
        # !!!!! 


    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        ex = WPManager.xr - robot.x
        ey = WPManager.yr - robot.y
        # !!!!! 
        # A COMPLETER EN TD : calcul de Vr
        Vr = k1 * np.sqrt(ex**2 + ey**2)       
        # !!!!! 

        # reference orientation
        # !!!!! 
        # A COMPLETER EN TD : calcul de thetar
        thetar = np.arctan2(ey, ex)
        # !!!!! 
        
        # !!!!!     
        # A COMPRENDRE EN TD : quelle est l'utilité des deux lignes de code suivantes ?
        #     (à conserver après le calcul de thetar)
        if math.fabs(robot.theta-thetar)>math.pi:
            thetar = thetar + math.copysign(2*math.pi,robot.theta)        
        # !!!!! 
        
        
    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):
        
        e_theta = thetar - robot.theta
        # angular velocity control input        
        # !!!!! 
        # A COMPLETER EN TD : calcul de omegar
        omegar = k2 * e_theta
        # !!!!! 

    
    # apply control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)    
    
    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted   
    simu.addData(robot, WPManager, Vr, thetar, omegar)
    
# end of loop on simulation time


# close all figures
plt.close("all")

# generate plots
simu.plotXY(1)
simu.plotXYTheta(2)
simu.plotVOmega(3)

#simu.runAnimation(WPManager.epsilonWP, 5)

# show plots
plt.show()





# Animation *********************************
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

robotBody, = ax.plot([], [], 'o-', lw=2)
robotDirection, = ax.plot([], [], '-', lw=1, color='k')
wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
WPArea, = ax.plot([], [], ':', lw=1, color='b')

thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
xWPArea = WPManager.epsilonWP*np.cos(thetaWPArea)
yWPArea = WPManager.epsilonWP*np.sin(thetaWPArea)

def initAnimation():
    robotDirection.set_data([], [])
    robotBody.set_data([], [])
    wayPoint.set_data([], [])
    WPArea.set_data([], [])
    robotBody.set_color('r')
    robotBody.set_markersize(20)    
    time_text.set_text('')
    return robotBody,robotDirection, wayPoint, time_text, WPArea  
    
def animate(i):  
    robotBody.set_data(simu.x[i], simu.y[i])          
    wayPoint.set_data(simu.xr[i], simu.yr[i])
    WPArea.set_data(simu.xr[i]+xWPArea.transpose(), simu.yr[i]+yWPArea.transpose())    
    thisx = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
    thisy = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
    robotDirection.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*simu.dt))
    return robotBody,robotDirection, wayPoint, time_text, WPArea

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t)),
    interval=4, blit=True, init_func=initAnimation, repeat=False)
#interval=25

#ani.save('robot.mp4', fps=15)

