from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as u
import math
import matplotlib.pyplot as plt
import numpy as np

pdb = PDBFile('1tki.pdb')
spring_constant=0.00050
pullto=np.zeros(3)
extension = []
t = []
spring_k = spring_constant * u.kilocalorie / (u.mole * u.angstrom * u.angstrom)
forcefield = ForceField('amber99sb.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1*angstrom)

integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

simulation.minimizeEnergy()

pullforce = CustomExternalForce('spring_k*(dx^2+dy^2+dz^2); \
                                  dx=abs(x-x0); \
                                  dy=abs(y-y0); \
                                  dz=abs(z-z0);')
pullforce.addPerParticleParameter("x0")
pullforce.addPerParticleParameter("y0")
pullforce.addPerParticleParameter("z0")
pullforce.addGlobalParameter("spring_k", spring_k)
pullforce.addParticle(5737, [pullto[0], pullto[1], pullto[2]])

system = simulation.context.getSystem()
system.addForce(pullforce)
currentState = simulation.context.getState()
simulation.context.reinitialize()
simulation.context.setState(currentState)
currentContext = simulation.context
vel = np.zeros((5738,3))
vel = (vel * u.angstrom)/u.picosecond
currentContext.setVelocities(vel)
system.setParticleMass(0,0)

simulation.context.setPositions(pdb.positions)

pos = simulation.context.getState(getPositions=True).getPositions()
b0_pos = np.array([pos[5737]/u.angstrom])
a0_pos = np.array([pos[0]/u.angstrom])
b_t = np.array([])
unit_vector = np.array([])
dist_init = np.linalg.norm(b0_pos-a0_pos)
F = []
W = np.array([])

for i in range(2000):
        # Get current state tu update the system
        pos = simulation.context.getState(getPositions=True).getPositions()   
        alpha = 1 + 0.02*i 
        b_t = pos[5737]/u.angstrom
        a_t = pos[0]/u.angstrom
        f_t = 1 + ((b0_pos-a0_pos)/dist_init)*alpha
        f_t = np.squeeze(f_t)
        dist_now = np.linalg.norm(b_t-f_t)
        distance = np.linalg.norm(b_t - a_t)
        # stdout.write('\r{0}'.format(pos[0]))
        stdout.write('\r{}'.format(i))
        a = (dist_now - 1)/dist_now
        force = spring_constant * a * (b_t - f_t)
        F.append(np.linalg.norm(force))
        # w = spring_constant*0.02*0.5(0.02*i*i - )
        
        unit_vector = np.array([pos[5737]/u.angstrom - pos[0]/u.angstrom])
        unit_vector = np.squeeze(unit_vector)
        dist = np.linalg.norm((pos[5737]/u.angstrom)-(pos[0]/u.angstrom))
        unit_vector = unit_vector/dist
        extension.append(dist)
        t.append(i)
        pullto[0] = (pos[5737][0]/u.angstrom)+0.8*(i+1)*unit_vector[0]
        pullto[1] = (pos[5737][1]/u.angstrom)+0.8*(i+1)*unit_vector[1]
        pullto[2] = (pos[5737][2]/u.angstrom)+0.8*(i+1)*unit_vector[2]
        pullforce.setParticleParameters(0,595,[pullto[0],pullto[1],pullto[2] ])
        pullforce.updateParametersInContext(currentContext)
        curr_vel = simulation.context.getState(getVelocities=True).getVelocities()
        curr_vel[0] = 0*(curr_vel[0])
        currContext = simulation.context
        currContext.setVelocities(curr_vel)
        # stdout.write('\r{0}'.format(simulation.context.getState(getVelocities=True).getVelocities()[0]))
        simulation.step(1)

positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))
plt.plot(extension,F)
plt.xlabel('Extension')
plt.ylabel('Force')
plt.show()
plt.plot(t,extension)
plt.show()


