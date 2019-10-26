from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as u
import math
import matplotlib.pyplot as plt
import numpy as np

beta = 0.0083144621
spring_constant=0.0050
pullto=np.zeros(3)
extension = []
t = []
spring_k = spring_constant * u.kilocalorie / (u.mole * u.angstrom * u.angstrom)
z=np.array([])
W = np.array([])
eta = np.array([])
velocity = 0.02    #nm/ps
time_step = 2e-15
delta_z = 0.05
h_l_label = np.array([])
h_l_val = np.array([])
U = np.array([])
K = 500


for k in range(K):
        pdb = PDBFile('villin.pdb')
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
        init_z = np.linalg.norm(b0_pos-a0_pos)
        temp_label = []
        temp_val = []
        u_ik = []
        print('value of k = {}' .format(k))
        for i in range(100000):
                prev_z = np.linalg.norm((pos[595]/u.angstrom)-pos[0]/u.angstrom)
                ext = prev_z-init_z
                u_ik.append(np.exp(-1*beta*spring_constant*0.5*((ext - (velocity*time_step*10))**2)))
                temp_z.append(ext)
                if(i==0):
                        temp += temp_z[i]
                else:
                        temp += temp_z[i] + temp_z[i-1]

                # print(int(ext/delta_z))
                w = spring_constant*velocity*0.5*(velocity*((10*time_step)**2) - temp)
                temp_label.append(int(ext/delta_z))
                temp_val.append(np.exp(-1*w*beta))
                
                temp_W.append(w)
                temp_eta.append(np.exp(-1*w*beta))
                alpha += velocity
                pos = simulation.context.getState(getPositions=True).getPositions()   


                b_t = pos[595]/u.angstrom
                a_t = pos[0]/u.angstrom
                c_t = ((b_t-a0_pos)/np.linalg.norm(b_t-a0_pos))*alpha + b_t
                c_t = np.squeeze(c_t)
                dist_now = np.linalg.norm(b_t-c_t)
                stdout.write('\r{}'.format(i))
                a = (dist_now - 1)/dist_now
                force = spring_constant * a * (b_t - c_t)
                F.append(np.linalg.norm(force))
                dist = np.linalg.norm((pos[595]/u.angstrom)-(pos[0]/u.angstrom))
                extension.append(dist)
                t.append(i)
                pullto = c_t
                pullforce.setParticleParameters(0,595,[pullto[0],pullto[1],pullto[2]])
                pullforce.updateParametersInContext(currentContext)
                simulation.step(10)
        if(k==0):
                z = np.array(temp_z)
                W = np.array(temp_W)
                eta = np.array(temp_eta)
                h_l_label = np.array(temp_label)
                h_l_val = np.array(temp_val)
                U = np.array(u_ik)
                positions = simulation.context.getState(getPositions=True).getPositions()
                PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))

        else:
                z = np.vstack((z,temp_z))
                W = np.vstack((W,temp_W))
                eta = np.vstack((eta,temp_eta))
                h_l_label = np.vstack((h_l_label,temp_label))
                h_l_val = np.vstack((h_l_val,temp_val))
                U = np.vstack((U,u_ik))

eta_final = np.mean(eta,axis=0)
U_final = np.mean(U,axis=0)

total_label = []
total_val =  []

for i in range(100000):
        a_label = []
        a_val = []
        col = h_l_label[:,i]
        for index,j in enumerate(col):
                if j not in a_label:
                        a_label.append(j)
                        a_val.append(h_l_val[index][i]/K)
                else:
                        try:
                                ind = a_label.index(j)
                        except ValueError:
                                pass
                        else:        
                                a_val[ind] += h_l_val[index][i]/K

        total_label.append(a_label)
        total_val.append(a_val)

total_label = np.array([total_label])
total_val = np.array([total_val])

den = 0
for i in range(100000):
       den += U_final[i]/eta_final[i]

i = -1
G_l = []
G_val = []

for cell in total_label.flat:
        i += 1
        for j in range(len(cell)):
                if cell[j] not in G_l:
                        G_l.append(cell[j])
                        G_val.append(total_val.flat[i][j]/eta_final[i])
                else:
                        try:
                                ind = G_l.index(cell[j])
                        except ValueError:
                                pass
                        else:        
                                G_val[ind] += total_val.flat[i][j]/eta_final[i]
G_l = np.array(G_l)
G_l = (G_l - 0.5)*delta_z
G_val = np.array(G_val)
G_val = G_val/den
G_val = beta*(-np.log(G_val))*300

plt.plot(G_l,G_val,'ro')
plt.xlabel('Extension')
plt.ylabel('G(z)')
# plt.show()
plt.savefig('plot.png', dpi=300)


