from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as u
import math
import matplotlib.pyplot as plt
import numpy as np

beta = 1
spring_constant=0.00050
pullto=np.zeros(3)
extension = []
t = []
spring_k = spring_constant * u.kilocalorie / (u.mole * u.angstrom * u.angstrom)
z=np.array([])
W = np.array([])
eta = np.array([])
velocity = 2    #Angstrom/ns
time_step = 2e-15



for k in range(10):
        pdb = PDBFile('input.pdb')
        forcefield = ForceField('amber99sb.xml', 'amber99_obc.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
        system.setParticleMass(0,0)
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        system = simulation.context.getSystem()
        pullforce = CustomExternalForce('spring_k*(dx^2+dy^2+dz^2); \
                                          dx=abs(x-x0); \
                                          dy=abs(y-y0); \
                                          dz=abs(z-z0);')
        pullforce.addPerParticleParameter("x0")
        pullforce.addPerParticleParameter("y0")
        pullforce.addPerParticleParameter("z0")
        pullforce.addGlobalParameter("spring_k", spring_k)
        pullforce.addParticle(595, [pullto[0], pullto[1], pullto[2]])

        system.addForce(pullforce)
        currentState = simulation.context.getState()
        simulation.context.reinitialize()
        simulation.context.setState(currentState)
        currentContext = simulation.context

        vel = np.zeros((596,3))
        vel = (vel * u.angstrom)/u.picosecond
        currentContext.setVelocities(vel)
        simulation.context.setPositions(pdb.positions)
        pos = simulation.context.getState(getPositions=True).getPositions()
        b0_pos = pos[595]/u.angstrom
        a0_pos = pos[0]/u.angstrom

        temp_z=[]
        temp_W = []
        temp_eta = []
        temp = 0
        alpha = 0
        F = []
        print(k)
        for i in range(200000):
                prev_b = pos[595]
                temp_z.append(velocity*time_step*(i+1))
                if(i==0):
                        temp += temp_z[i]
                else:
                        temp += temp_z[i] + temp_z[i-1]

                w = spring_constant*velocity*0.5*(velocity*(time_step**2) - temp)
                temp_W.append(w)
                temp_eta.append(np.exp(-1*w*beta))
                alpha += velocity
                pos = simulation.context.getState(getPositions=True).getPositions()   
                b_t = pos[595]/u.angstrom
                # print(np.linalg.norm((b_t*u.nanometer)-prev_b))
                # print(velocity*time_step*(i+1)*1e9,end='\n')
                # print('\n')

                a_t = pos[0]/u.angstrom
                c_t = ((b0_pos-a0_pos)/np.linalg.norm(b0_pos-a0_pos))*alpha + b0_pos
                c_t = np.squeeze(c_t)
                dist_now = np.linalg.norm(b_t-c_t)
                stdout.write('\r{}'.format(i))
                a = (dist_now - 1)/dist_now
                force = spring_constant * a * (b_t - c_t)
                F.append(np.linalg.norm(force))
                # w = spring_constant*vel*0.5(vel*(i*0.002*1e-12)**2 -(0.002*1e-12)*() )
                dist = np.linalg.norm((pos[595]/u.angstrom)-(pos[0]/u.angstrom))
                extension.append(dist)
                t.append(i)
                pullto = c_t
                pullforce.setParticleParameters(0,595,[pullto[0],pullto[1],pullto[2]])
                pullforce.updateParametersInContext(currentContext)
                simulation.step(100)
        if(k==0):
                z = np.array(temp_z)
                W = np.array(temp_W)
                eta = np.array(temp_eta)
        else:
                z = np.vstack((z,temp_z))
                W = np.vstack((W,temp_W))
                eta = np.vstack((eta,temp_eta))
        
print(eta.shape)
eta_final = np.mean(eta,axis=1)
print(eta_final.shape)


positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))
# plt.plot(extension,F)
# plt.xlabel('Extension')
# plt.ylabel('Force')
# plt.show()
# plt.plot(t,extension)
# plt.ylabel('Extension')
# plt.xlabel('Time')
# plt.show()


