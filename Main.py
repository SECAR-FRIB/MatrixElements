import math
import array
import random
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.special import rel_entr

import cosyHelp as cosy
import plotHelp as plot

#==============================================================================================================
#============================================ INPUT -- MODIFY HERE ============================================
#==============================================================================================================

# Distribution at target
xh = 0 # center of x distribution in mm
yh = 0 # center of x distribution in mm
widthX = 1.5/2 # half x- beam spot in mm
widthY = 1.5/2 # half y- beam spot in mm
dE = 0.0025 # half of total energy spread in fraction 1% = 0.01
#dE = 0.00 # half of total energy spread in fraction 1% = 0.01
aX = 10 # mrad
aY = 15 # mrad

FP = 'FP2' # Focal plane to consider

numberMC = 100 # number of beam particles. Was 800

VD1542w = 1.77*25.4 #mm
VD1542h = 2.5/np.sqrt(2)*25.4 #mm
VD1638w = 4.951*25.4 #mm
VD1638h = 3.134/np.sqrt(2)*25.4 #mm

# Q2 - Q3 additional scaling
# FP1:
#magnets = [[0.8, 0.85], [0.8, 0.9], [0.8, 1.0], [0.8, 1.1], \
#		   [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], \
#		   [1.2, 0.85], [1.2, 0.9], [1.4, 1.0], [1.4, 1.1]]
# FP2: M11, M12
magnets = [[0.8, 0.95], [0.8, 1.0], [0.8, 1.1], [0.8, 1.2], \
		   [1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], \
		   [1.2, 0.85], [1.2, 0.9], [1.2, 0.95], [1.2, 1.0]]
# FP2: M18
#magnets = [[1.0, 0.85], [1.0, 0.9], [1.0, 1.0], [1.0, 1.1], \
#		   [1.3, 0.85], [1.3, 0.9], [1.3, 0.95], [1.3, 1.0], \
#		   [1.7, 0.85], [1.7, 0.9], [1.7, 0.95], [1.7, 1.0]]

#=============================================================================================================

if FP == 'FP1':
	xmin = -VD1542w/2-7
	xmax = VD1542w/2-7
	ymin = -VD1542h/2
	ymax = VD1542h/2
elif FP == 'FP2':
	xmin = -VD1638w/2
	xmax = VD1638w/2
	ymin = -VD1638h/2
	ymax = VD1638h/2
dimVD = [xmin, xmax, ymin, ymax, FP]

#=============================================================================================================

# Function that generates positions and angles at the target within the input parameters
def generateInitialDistribution(widthX, widthY, aX, aY):
	beam = np.zeros([numberMC, 8], dtype="float64", order="F")

	for j in range(numberMC):
		# Sampling within ellipse of possible positions
		r = widthX * np.sqrt(random.uniform(0, 1))
		theta = random.uniform(0, 1) * 2 * np.pi
		x = xh + r * np.cos(theta)
		y = yh + widthY/widthX *  r * np.sin(theta)
		beam[j][0] = x/1000 # in m
		beam[j][2] = y/1000 # in m

		# Sampling within ellipse of possible angles
		r = aX * np.sqrt(random.uniform(0, 1))
		theta = random.uniform(0, 1) * 2 * np.pi
		angleX = r * np.cos(theta)
		if aX != 0:
			angleY = aY/aX *  r * np.sin(theta)
		else:
			angleY = aY * np.sqrt(random.uniform(0, 1))
		beam[j][1] = angleX/1000 # in rad
		beam[j][3] = angleY/1000 # in rad
		beam[j][5] = dE
	return (beam)

# Function that generates KDE. It has an option of creating a reference KDE
def generateKDE(saveReference = False, coeff = [], beam0 = generateInitialDistribution(widthX, widthY, aX, aY)):
	storeKDE = []
	storePOS = []

	for i in range(len(magnets)):

		beamFinal = cosy.transportTotal(beam0, i, coeff)
		x = beamFinal[:, 0]*1000  # in mm
		y = beamFinal[:, 2]*1000  # in mm
		values_ = np.vstack([x, y])
		kernel_ = gaussian_kde(values_)
		x_min = dimVD[0]
		x_max = dimVD[1]
		y_min = dimVD[2]
		y_max = dimVD[3]
		X_, Y_ = np.mgrid[x_min:x_max:170j, y_min:y_max:130j]   # change here number of pixels
		positions_ = np.vstack([X_.ravel(), Y_.ravel()])
		storeKDE.append(kernel_(positions_))
		storePOS.append(x)
		storePOS.append(y)

	KDE = pd.DataFrame(storeKDE)
	POS = pd.DataFrame(storePOS)
	if (saveReference):
		KDE.to_csv(r"referenceKDE.csv", header=True, index=False)
		POS.to_csv(r"referencePOS.csv", header=True, index=False)
	return( KDE, POS )

# Function that compares a generated KDE against a reference KDE using a KL metric
def compareKDE(coeff = [], beam0 = generateInitialDistribution(widthX, widthY, aX, aY)):

	KDE, POS = generateKDE(coeff = coeff, beam0 = beam0)
	#plotDistribution( KDE, POS )
	KDEReference = pd.read_csv(r"referenceKDE.csv", header = 0)
	#POSReference = pd.read_csv(r"referencePOS.csv", header = 0)
	#plotDistribution( KDEReference, POSReference )
	KL = 0
	for i in range(1):
	#for i in range(len(magnets)):
		P = np.asarray(KDE.iloc[i,:])
		Q = np.asarray(KDEReference.iloc[i,:])
		for j in range(len(P)):   # Putting a lower limit in the distribution
			if P[j] < 1e-150:
				P[j] = 1e-150
			if Q[j] < 1e-150:
				Q[j] = 1e-150
		KL = KL + sum(rel_entr(P, Q))
	return ( KL )

# Function that calculates KL (compared with reference distributions) for a grid of matrix elements
def Grid1DKL(beam0 = generateInitialDistribution(widthX, widthY, aX, aY)):
	KL = []
	#mEl = np.linspace(0.5, 2.0, 16)  # m11 : 1.294853909271475
	#mEl = np.arange(0.05, 0.5, 0.02)  # m12 : 0.2687314930972021
	#mEl = np.arange(-1.0, -0.5, 0.05)  # m18 : -.6983777753892170E-01
	#mEl = np.arange(2.0, 3.0, 0.05)  # m21 : 2.517906787885651
	mEl = np.arange(0.5, 0.92, 0.02)  # m33 : 0.7296696849107546
	for el in mEl:
			temp = compareKDE( coeff = [[2, 3, el]], beam0 = beam0 )    # Change the coeff pos. accordingly!
			KL.append([el, temp])
			print(el, temp)
	print(KL)
	dataframe = pd.DataFrame(KL)
	dataframe.to_csv(r"GridKL.csv")

def Grid2DKL(beam0 = generateInitialDistribution(widthX, widthY, aX, aY)):
	KL = []
	mEl1 = np.linspace(0.5, 2.0, 16)  # m11 : 1.294853909271475
	#mEl1 = np.arange(0.05, 0.5, 0.02)  # m12 : 0.2687314930972021
	#mEl2 = np.arange(2.0, 3.0, 0.05)  # m21 : 2.517906787885651
	mEl2 = np.arange(0.5, 0.92, 0.02)  # m33 : 0.7296696849107546
	for el1 in mEl1:
		for el2 in mEl2:
			temp = compareKDE( coeff = [[0, 1, el1], [1, 2, el2]], beam0 = beam0 )
			KL.append([el1, el2, temp])
			print(el1, el2, temp)
	print(KL)
	dataframe = pd.DataFrame(KL)
	dataframe.to_csv(r"GridKL.csv")

#=============================================================================================================
# Main program
#=============================================================================================================

if __name__ == '__main__':

	print('Start running')

	#cosy.COSYSingleRun(magnets, FP)     # Testing of a single cosy run
	#cosy.generateMatrices(magnets, FP)    # Command to generate matrices for all magnet settings

	beam0 = generateInitialDistribution(widthX, widthY, aX, aY)
	#KDE, POS =  generateKDE(saveReference = True, beam0 = beam0)
	#KDE, POS =  generateKDE(coeff = [[0, 1, 2.0]], beam0 = beam0)
	#plot.plotDistribution( KDE, POS, magnets = magnets, dimVD = dimVD )    # Displays KDE
	#plot.plotDistribution( magnets = magnets, dimVD = dimVD )    # Displays KDE
	#print ( compareKDE( coeff = [[0, 1, 1.294853909271475]], beam0 = beam0 ) )
	#Grid1DKL(beam0)
	#plot.plot1DKL('M33')
	#Grid2DKL(beam0)
	plot.plot2DKL()

	# KL_minimize()
