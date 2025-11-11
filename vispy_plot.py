import numpy as np
import numba as nb

import vispy.app as va
import vispy.color as vc
import vispy.plot as vp
import vispy.scene as vs

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



    fig = vp.Fig(size=(800, 600))

    view = fig.central_widget.add_view()
    view.camera = vs.TurntableCamera(up='z', fov=60)

    tax = vs.Axis(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(0, -100),
                 font_size=16, axis_color='k', tick_color='k', text_color='k',
                 parent=view.scene, domain=(0, TIME))
    tax.transform = vs.STTransform(translate=(0, 0, -0.2))

    yax = vs.Axis(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(-100, 0),
                    font_size=16, axis_color='k', tick_color='k', text_color='k',
                    parent=view.scene, domain=(0, STOCK))
    yax.transform = vs.STTransform(translate=(0, 0, -0.2))
    
    cmap = vc.get_colormap("coolwarm")

    plot = vs.SurfacePlot(z=optionGrid, x=timeGrid, y=stockGrid)

    plot.transform = vs.transforms.MatrixTransform()
    plot.transform.scale([1 / STOCK, 1 / TIME, 1 / STOCK])
    plot.transform.translate([-0.5, -0.5, 0])
    plot.mesh_data.set_vertex_colors(cmap.map(optionGrid))

    view.add(plot)

    def update(_):
        if statusFlag[0]:
            plot.set_data(timeGrid, stockGrid, optionGrid)
            plot.mesh_data.set_vertex_colors(cmap.map(optionGrid / np.max(optionGrid)))

    timer = va.Timer(interval=1 / UPDATE_FREQUENCY, connect=update, start=True) 
    fig.show(visible=True, run=True)

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


