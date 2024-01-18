
from CircularScatterer2D import AcousticSc_AnalyticalSol_Cylinder2D

#_______________________________ Main body _____________________________________
if __name__ == "__main__":
    r_s = 0.05   #scatterer radius
    p0 = 1       #incident pressure amplitude
    freq = 1000  #Hz  
    obj1 = AcousticSc_AnalyticalSol_Cylinder2D(r_s, p0, freq)
    ps = obj1.scattered_pressure_plots()   #scattered pressure plots
    HE = obj1.Helmholtz_eqn_calculation(plot=False); #Helmholtz equation evaluation
