# -*- coding: utf-8 -*-
"""
(c) author: S. Bertrand
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import AStar
import Robot as rob
import Timer as tmr

    
dimX = 20
dimY = 10
    
occupancyGrid = np.zeros([dimX, dimY])
occupancyGrid[2:4,2:4]=1
occupancyGrid[6,1:3]=1
occupancyGrid[4:7,3]=1
occupancyGrid[8:10,2:6]=1
occupancyGrid[4:6,6:8]=1
occupancyGrid[6,5:7]=1
occupancyGrid[10:12,7:10]=1
occupancyGrid[11,0:3]=1
occupancyGrid[11:15,4:6]=1
occupancyGrid[13:15,2:5]=1
occupancyGrid[15,3:5]=1
occupancyGrid[16:18,3]=1
occupancyGrid[18:20,5]=1
occupancyGrid[14,7:9]=1
occupancyGrid[16,7:10]=1


adjacency = 4

carte = AStar.Map(dimX,dimY, adjacency)
carte.initCoordinates()
carte.loadOccupancy(occupancyGrid)
carte.generateGraph()
carte.plot(1)

  
epsilon = 1


print("A* algorithm running ...")
closedList, successFlag = carte.AStarFindPath(0,199, epsilon)
#print(closedList)
if (successFlag==True):
    print("  - A* terminated with success: path found")

    path = carte.builtPath(closedList)
    print("path:", path)
    waypoints = [(node % dimX, node // dimX) for node in path]
    print("waypoints :", waypoints)

    carte.plotPathOnMap(path, 2)
    carte.plotExploredTree(closedList, 3)
else :
    print("  - A* failed: no path found")
    waypoints = []
     
# Afficher la carte et le chemin avec la méthode existante `plotPathOnMap`
carte.plotPathOnMap(path, 1)  # Afficher la carte avec le chemin

# Animation du robot
x0, y0, theta0 = 0.0, 0.0, 0.0
robot = rob.Robot(x0, y0, theta0)
simu = rob.RobotSimulation(robot, 0.0, 50.0, 0.01)
WPlist = waypoints  # Utilisation des waypoints trouvés par A*
WPManager = rob.WPManager(WPlist, epsilonWP=0.2)

# Initialisation des variables de contrôle
Vr, thetar, omegar = 0.0, 0.0, 0.0
k1, k2 = 1.0, 20.0
timerPositionCtrl = tmr.Timer(0.2)
timerOrientationCtrl = tmr.Timer(0.05)

# Configuration de l'animation
robotBody, = plt.plot([], [], 'ro-', lw=2)
robotDirection, = plt.plot([], [], '-', lw=1, color='k')
wayPoint, = plt.plot([], [], 'bo-', lw=2)
time_template = 'time = %.1fs'
time_text = plt.text(0.05, 0.9, '', transform=plt.gca().transAxes)
WPArea, = plt.plot([], [], ':', lw=1, color='b')

thetaWPArea = np.arange(0.0, 2.0 * math.pi + 2 * math.pi / 30.0, 2.0 * math.pi / 30.0)
xWPArea = WPManager.epsilonWP * np.cos(thetaWPArea)
yWPArea = WPManager.epsilonWP * np.sin(thetaWPArea)

# Données pour stocker la position du robot au cours du temps
positions = []

def initAnimation():
    robotDirection.set_data([], [])
    robotBody.set_data([], [])
    wayPoint.set_data([], [])
    WPArea.set_data([], [])
    time_text.set_text('')
    return robotBody, robotDirection, wayPoint, time_text, WPArea

def animate(i):
    global Vr, thetar, omegar  # Pour pouvoir modifier les variables globales

    # Vérification et mise à jour du waypoint actuel
    if WPManager.distanceToCurrentWP(robot.x, robot.y) < WPManager.epsilonWP:
        WPManager.switchToNextWP()

    # Calcul du contrôle de position
    if timerPositionCtrl.isEllapsed(i * simu.dt):
        ex = WPManager.xr - robot.x
        ey = WPManager.yr - robot.y
        Vr = k1 * np.sqrt(ex ** 2 + ey ** 2)  # Commande de vitesse linéaire
        thetar = np.arctan2(ey, ex)  # Orientation de référence

        # Correction de l'orientation si elle dépasse pi
        if math.fabs(robot.theta - thetar) > math.pi:
            thetar += math.copysign(2 * math.pi, robot.theta)

    # Calcul du contrôle d'orientation
    if timerOrientationCtrl.isEllapsed(i * simu.dt):
        e_theta = thetar - robot.theta
        omegar = k2 * e_theta  # Commande de vitesse angulaire

    # Appliquer les contrôles au robot
    robot.setV(Vr)
    robot.setOmega(omegar)
    
    # Intégrer le mouvement
    robot.integrateMotion(simu.dt)

    # Stocker la position du robot pour l'animation
    positions.append((robot.x, robot.y))

    # Mettre à jour les données pour l'animation
    robotBody.set_data([robot.x], [robot.y])
    wayPoint.set_data([WPManager.xr], [WPManager.yr])
    WPArea.set_data(WPManager.xr + xWPArea.transpose(), WPManager.yr + yWPArea.transpose())
    
    thisx = [robot.x, robot.x + 0.5 * math.cos(robot.theta)]
    thisy = [robot.y, robot.y + 0.5 * math.sin(robot.theta)]
    robotDirection.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * simu.dt))

    return robotBody, robotDirection, wayPoint, time_text, WPArea

# Lancer l'animation sur la carte tracée par A*
ani = animation.FuncAnimation(plt.gcf(), animate, frames=np.arange(0, len(simu.t)),
                              interval=4, blit=True, init_func=initAnimation, repeat=False)


plt.show()



