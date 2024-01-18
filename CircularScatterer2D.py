
__version__ = '1.2'
__author__  = "Siddharth Nair (nair40@purdue.edu)"
__date__    = '2022-May-13'

__doc__     ='''

This code defines a class AcousticSc_analyticaSol_cylinder2D() that evaluates 
the analytical solution of acoustic scattering with a circular scatterer
(cylindrical shape in infinite medium) in a 2D domain.
The scattered pressure is calculated in polar coordinates and plotted in a 
[0,1]x[0,1] square domain. The class evaluates the solutions based on the 
following user inputs - radius of scatterer (r_s), incident pressure 
amplitude (p0), and frequency (freq).

Directly executing this default code will evaluate the analytical solution (ps),
plot the scattered field, and evaluate the Helmholtz equation (HE) 
for r_s = 0.05 m, p0 = 1. Pa, and freq = 1000 Hz.

The use of function definitions used inside this class is detailed below-
1. def scattererd_pressure_calculation(): 
   This function is responsible for evaluating the analytical solution in 
   polar coordinates. The analytical solution is
                      __ 
     p_s(r, theta) =  \    A_n*H_n^(1)(k*r)*cos(n*theta)   
                      /__

2. def Helmholtz_eqn_calculation(): 
   This function calculates and plots (optional- to verify that the solution is 
   close to zero) the Helmholtz equation satisfied by the analytical solution
   evaluated in scattererd_pressure_calculation(). 
   Note that this is not evaluated using a numerical gradient functions, rather
   the author directly evaluates the following Helmholtz eqution satisified by 
   the symbolic analytical solution ps above

          d^2p_s       1  dp_s       1   d^2p_s
          ------   +  --- ----   +  --- --------   +   k^2 p_s = 0
           dr^2        r   dr       r^2 dtheta^2
   
   The author used mathematica to find the symbolic derivatives for this task.

3. def scattered_pressure_plots(): 
   This function plots the real, imaginary and absolute values of the analytical
   solution obtained in scattererd_pressure_calculation() within a square domain 
   of size [0,1]x[0,1].

@endofdocs
'''

import numpy as np
import scipy.special as sp
import math
import matplotlib.pyplot as plt
# %matplotlib inline


# ______________ Begin class AcousticSc_AnalyticalSol_cylinder2D() _____________

class AcousticSc_AnalyticalSol_Cylinder2D():
    def __init__(self, r_s, p0=1., freq=500):
        """
        r_s          =  circular scatterer radius in 'm'
        p0           =  Incident pressure amplitude in 'Pa'. Default value is 1 Pa.
        freq         =  frequency of the scattered wave (in a homogeneous medium) in 'Hz'.
                        Default value is 500 Hz.
        n_r, n_theta =  number of r and theta values evaluated. These define the 
                        resolution of the plots. Here, n_r=200 and n_theta=200                        
        cs           =  speed of sound in air - 342.21 m/s
        k            =  wavenumber inside the medium in '1/m'. Calculated based on 
                        freq and cs.
        n_max        =  Max number of harmonics used to calculate the anaytical solution.     
        """
        self.r_s = r_s #scatterer radius
        self.p0 = p0
        self.f = freq #Hz
        self.n_r = 200
        self.n_theta = 200        
        self.cs = 342.21 #m/s
        self.k = 2*math.pi*self.f/self.cs  #wavenumber
        self.n_max = 25
        self.r = np.linspace(1.2*self.r_s, 1.0, self.n_r)
        self.theta = np.linspace(0, 2*math.pi, self.n_theta)

    def scattered_pressure_calculation(self):
        """
        Evaluates the analytical scattered pressure field p_s in 'Pa'.
        """
        n_r = self.n_r
        n_theta = self.n_theta
        n_max = self.n_max
        r_s = self.r_s
        k = self.k
        ka = k*r_s
        r = self.r
        theta = self.theta
        n = np.linspace(1,n_max,n_max)
        p0 = self.p0

        #_______________ calculating the constants: A0 and An __________________
        a0 = -p0*sp.jv(1, ka)/sp.hankel1(1, ka)
        an = -2*(1j)**(n)*p0*0.5*(sp.jv(n-1, ka) - sp.jv(n+1, ka))/(n*sp.hankel1(n, ka)/ka - sp.hankel1(n+1, ka))

        #An -> [n_r, n_theta, n_max+1]: repeat the a0, an for all r and theta values
        An = np.repeat(np.repeat(np.concatenate((a0.reshape(1), an), axis=0).reshape(1,1,n_max+1), n_r, axis=0),\
                      n_theta, axis=1) 
        N = np.repeat(np.repeat(np.linspace(0,n_max,n_max+1).reshape(1, 1, n_max+1), n_r, axis=0),\
                      n_theta, axis=1)
        R = np.repeat(np.repeat(r.reshape(n_r, 1, 1), n_theta, axis=1), n_max+1, axis=2)
        Theta = np.repeat(np.repeat(theta.reshape(1, n_theta, 1), n_r, axis=0), n_max+1, axis=2)
        ps = np.sum(An*sp.hankel1(N, k*R)*np.cos(N*Theta), axis=2)

        return ps

    def Helmholtz_eqn_calculation(self, plot=False):
        """
        Evaluates the Helmholtz equation satisfied by the analytical scattered 
        pressure field p_s.
        """      
        n_r = self.n_r
        n_theta = self.n_theta
        n_max = self.n_max
        r_s = self.r_s
        k = self.k
        ka = k*r_s
        r = self.r
        theta = self.theta
        n = np.linspace(2,n_max,n_max-1)
        p0 = self.p0

        #_______________ calculating the constants: A0 and An __________________
        a0 = -p0*sp.jv(1, ka)/sp.hankel1(1, ka)
        an = -2*(1j)**(n)*p0*0.5*(sp.jv(n-1, ka) - sp.jv(n+1, ka))/(n*sp.hankel1(n, ka)/ka - sp.hankel1(n+1, ka))

        #An -> [n_r, n_theta, n_max+1]: repeat the a0, an for all r and theta values
        An = np.repeat(np.repeat(an.reshape(1,1,n_max-1), n_r, axis=0),\
                      n_theta, axis=1) 
        N = np.repeat(np.repeat(np.linspace(2,n_max,n_max-1).reshape(1, 1, n_max-1), n_r, axis=0),\
                      n_theta, axis=1)
        R = np.repeat(np.repeat(r.reshape(n_r, 1, 1), n_theta, axis=1), n_max-1, axis=2)
        Theta = np.repeat(np.repeat(theta.reshape(1, n_theta, 1), n_r, axis=0), n_max-1, axis=2)

        #_____________________Satisfying the Helmholtz equation_________________
        #at n=0
        HE0 = a0*(k**2)*(-sp.hankel1(0,k*R) + 
                         (1/(k*R))*sp.hankel1(1, k*R)) - a0*(k/R)*sp.hankel1(1, k*R) + a0*k**2*sp.hankel1(0, k*R)                      

        #at n=1
        a1 = -2*(1j)**(1)*p0*0.5*(sp.jv(0, ka) - sp.jv(2, ka))/(1*sp.hankel1(1, ka)/ka - sp.hankel1(2, ka))

        d2pdr2_1 = a1*k**2*np.cos(1*Theta)*((-1/(k*R))*sp.hankel1(0, k*R) +
                                            ((2- (k*R)**2)/(k*R)**2)*sp.hankel1(1, k*R))
        dpdr_1 = a1*np.cos(1*Theta)*(k/(2*R))*(sp.hankel1(0, k*R) - sp.hankel1(2, k*R))
        d2pdt2_1 = -a1*np.cos(1*Theta)*sp.hankel1(1, k*R)/(R**2)
        HE1 = d2pdr2_1 + dpdr_1 + d2pdt2_1 + a1*k**2*np.cos(1*Theta)*sp.hankel1(1, k*R)

        #for all n>1
        d2pdr2_n = k**2*((-1/(k*R))*sp.hankel1(N-1, k*R) + 
                         ((N + N**2 - (k*R)**2)/(k*R)**2)*sp.hankel1(N, k*R))
        dpdr_n = (k/(2*R))*(sp.hankel1(N-1, k*R) - sp.hankel1(N+1, k*R))
        d2pdt2_n = -(N/R)**2*sp.hankel1(N, k*R)
        k2p_n = k**2*sp.hankel1(N, k*R)

        helmholtz_eqn_value = np.sum(HE0 + HE1 + (An*np.cos(N*Theta)*(d2pdr2_n + dpdr_n + d2pdt2_n + k2p_n)), axis=2)
        
        #_________Visualize the helmholtz equation field over the domain________
        if plot:
            ps = self.scattered_pressure_calculation()

            x = np.repeat(self.r.reshape(self.n_r,1), self.n_theta, axis=1)\
            *np.cos(np.repeat(self.theta.reshape(1,self.n_theta),self.n_r, axis=0)) + 0.5
            y = np.repeat(self.r.reshape(self.n_r,1), self.n_theta, axis=1)\
            *np.sin(np.repeat(self.theta.reshape(1,self.n_theta), self.n_r, axis=0)) + 0.5    

            fig, ax1 = plt.subplots()
            plt.jet()
            c1=ax1.scatter(x, y, c=np.real(helmholtz_eqn_value))
            ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
            ax1.set_xlim([0,1])
            ax1.set_ylim([0,1])
            plt.title('Helmholtz equation- Re at f=%d Hz'%(self.f))
            plt.colorbar(c1)
            plt.show()    

            fig, ax1 = plt.subplots()
            plt.jet()
            c1=ax1.scatter(x, y, c=np.imag(helmholtz_eqn_value))
            ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
            ax1.set_xlim([0,1])
            ax1.set_ylim([0,1])
            plt.title('Helmholtz equation- Im at f=%d Hz'%(self.f))
            plt.colorbar(c1)
            plt.show()    

            fig, ax1 = plt.subplots()
            plt.jet()
            c1=ax1.scatter(x, y, c=np.abs(helmholtz_eqn_value))
            ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
            ax1.set_xlim([0,1])
            ax1.set_ylim([0,1])
            plt.title('Helmholtz equation- Abs at f=%d Hz'%(self.f))
            plt.colorbar(c1)
            plt.show()     
        
        return helmholtz_eqn_value  


    def scattered_pressure_plots(self):
        
        ps = self.scattered_pressure_calculation()

        x = np.repeat(self.r.reshape(self.n_r,1), self.n_theta, axis=1)\
        *np.cos(np.repeat(self.theta.reshape(1,self.n_theta),self.n_r, axis=0)) + 0.5
        y = np.repeat(self.r.reshape(self.n_r,1), self.n_theta, axis=1)\
        *np.sin(np.repeat(self.theta.reshape(1,self.n_theta), self.n_r, axis=0)) + 0.5             

        #_________________________________ Plots _______________________________
        fig, ax1 = plt.subplots()
        plt.jet()
        c1=ax1.scatter(x, y, c=np.real(ps))
        ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        plt.title('Analytical solution- Re(Ps) at f=%d Hz'%(self.f))
        plt.colorbar(c1)
        plt.show()    

        fig, ax1 = plt.subplots()
        plt.jet()
        c1=ax1.scatter(x, y, c=np.imag(ps))
        ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        plt.title('Analytical solution- Im(Ps) at f=%d Hz'%(self.f))
        plt.colorbar(c1)
        plt.show()           

        fig, ax1 = plt.subplots()
        plt.jet()
        c1=ax1.scatter(x, y, c=np.abs(ps))
        ax1.plot(self.r_s*np.cos(self.theta)+0.5, self.r_s*np.sin(self.theta)+0.5, 'k.')
        ax1.set_xlim([0,1])
        ax1.set_ylim([0,1])
        plt.title(' Analytical solution- Abs(Ps) at f=%d Hz'%(self.f))
        plt.colorbar(c1)
        plt.show()

# ______________ End of class AcousticSc_AnalyticalSol_cylinder2D() ____________
