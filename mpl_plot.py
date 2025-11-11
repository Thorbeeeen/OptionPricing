import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import multiprocessing as mp
import multiprocessing.shared_memory as sm

from RKDP_Coeff import *



TIME = 100
DISC_TIME = 0.1
STOCK = 100
DISC_STOCK = 1

VOLATILITY = 0.1
RF_INTEREST = 0

STRIKE_PRICE = 50

PLOTTING_CUTOFF = 100
UPDATE_FREQUENCY = 1
EVALUATIONS = 7
ERROR_LIMIT = 10e-10

timeAxis = np.arange(0, TIME, DISC_TIME, dtype=np.float64)
stockAxis = np.arange(0, STOCK, DISC_STOCK, dtype=np.float64)

TIME_POINTS = timeAxis.shape[0]
STOCK_POINTS = stockAxis.shape[0]

stockGrid, timeGrid = np.meshgrid(timeAxis, stockAxis, indexing="ij")

optionPayoff = np.maximum(stockAxis - STRIKE_PRICE, 0)



def plotting(optionGridName, statusFlagName):
    optionGridBuffer = sm.SharedMemory(name=optionGridName)
    optionGrid = np.ndarray((TIME_POINTS, STOCK_POINTS), 
                            dtype=np.float64, 
                            buffer=optionGridBuffer.buf)
    
    statusFlagBuffer = sm.SharedMemory(name=statusFlagName)
    statusFlag = np.ndarray((1,), 
                            dtype=np.int8,
                            buffer=statusFlagBuffer.buf)

    f, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

    def update(_):
        if statusFlag[0]:
            ax.clear()
            ax.plot_surface(timeGrid[:, 0:PLOTTING_CUTOFF], stockGrid[:, 0:PLOTTING_CUTOFF], optionGrid[:, 0:PLOTTING_CUTOFF], rcount=30, ccount=30, cmap='viridis')
            ax.set_xlabel('Stock Price')
            ax.set_ylabel('Time to Maturity')
            statusFlag[0] = 0
            

    a = ani.FuncAnimation(f, update, interval=1000 / UPDATE_FREQUENCY, cache_frame_data=False)
    plt.show()

    optionGridBuffer.close()
    statusFlagBuffer.close()



def simulation(optionGridName, statusFlagName):
    optionGridBuffer = sm.SharedMemory(name=optionGridName)
    optionGrid = np.ndarray((TIME_POINTS, STOCK_POINTS), 
                            dtype=np.float64, 
                            buffer=optionGridBuffer.buf)
    
    statusFlagBuffer = sm.SharedMemory(name=statusFlagName)
    statusFlag = np.ndarray((1,), 
                            dtype=np.int8,
                            buffer=statusFlagBuffer.buf)

    currentTime = 0
    currentStep = DISC_TIME
    currentState = optionPayoff 

    @nb.njit()
    def jitFunc(currentTime, currentStep, currentState, optionGrid, statusFlag):
        for i in range(TIME_POINTS):
            while currentTime < DISC_TIME * i:
                k = np.zeros((EVALUATIONS, STOCK_POINTS), dtype=np.float64)

                for j in range(EVALUATIONS):
                    evalPoint = currentState[:]

                    for l in range(j):
                        evalPoint += A[j, l] * k[l]
                    
                    for l in range(1, STOCK_POINTS - 1):
                        k[j, l] = (currentStep * (0.5 * (evalPoint[l + 1] - 2 * evalPoint[l] + evalPoint[l - 1]) * VOLATILITY ** 2 * stockAxis[l] ** 2 / (DISC_STOCK ** 2) 
                                - 0.5 * (evalPoint[l + 1] - evalPoint[l - 1]) * RF_INTEREST * stockAxis[l] / DISC_STOCK
                                + RF_INTEREST * evalPoint[l]))

                errorArray = np.zeros(STOCK_POINTS, dtype=np.float64)

                for j in range(EVALUATIONS):
                    errorArray += D[j] * k[j]

                errorValue = np.linalg.norm(errorArray)
                currentStep = 0.9 * currentStep * (ERROR_LIMIT / errorValue) ** 0.2

                if ERROR_LIMIT > errorValue:
                    currentTime += currentStep

                    for j in range(EVALUATIONS):
                        currentState += B5[j] * k[j]

            optionGrid[i] = currentState
            statusFlag[0] = 1

    jitFunc(currentTime, currentStep, currentState, optionGrid, statusFlag)        

    optionGridBuffer.close()
    statusFlagBuffer.close()



def main(): 
    optionGrid = sm.SharedMemory(create=True, size=timeGrid.nbytes)
    statusFlag = sm.SharedMemory(create=True, size=1)

    plottingProcess = mp.Process(target=plotting, name="Plotting", args=(optionGrid.name, statusFlag.name))
    simulationProcess = mp.Process(target=simulation, name="Simulation", args=(optionGrid.name, statusFlag.name))

    plottingProcess.start()
    simulationProcess.start()

    plottingProcess.join()
    simulationProcess.terminate()



if __name__ == "__main__":
    main()


