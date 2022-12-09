import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Function that displays KDE
def plotDistribution(KDE = pd.DataFrame(), POS = pd.DataFrame(), magnets = [], dimVD = []):

    if KDE.empty:
        KDE = pd.read_csv(r"referenceKDE.csv", header = 0)
        POS = pd.read_csv(r"referencePOS.csv", header = 0)

    nuRow = 3
    nuCol = 4
    fig, axs = plt.subplots(nuRow, nuCol, figsize=(17,9))

    for i in range(len(magnets)): # Reference plot can only include up to certain number of magnet combinations
        if i > nuCol*nuRow-1:
            j = nuCol*nuRow-1
        else:
            j = i

        x = POS.iloc[2*i,:]
        y = POS.iloc[2*i + 1,:]
        axs[int(j/nuCol), j%nuCol].scatter(x,y)
        axs[int(j/nuCol), j%nuCol].set_title(dimVD[4] + ' with Q2 = ' + str(magnets[i][0]) + ", Q3 = " + str(magnets[i][1]))
        x_min = dimVD[0]
        x_max = dimVD[1]
        y_min = dimVD[2]
        y_max = dimVD[3]
        kernel = np.asarray(KDE.iloc[i,:])
        Z_ = np.reshape(kernel.T, [170,130])       # change here number of pixels [170, 130]
        axs[int(j/nuCol), j%nuCol].imshow(np.rot90(Z_), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
        axs[int(j/nuCol), j%nuCol].set_xlabel('x[mm]')
        axs[int(j/nuCol), j%nuCol].set_ylabel('y[mm]')
    plt.tight_layout()
    plt.savefig("Positions.pdf")
    plt.show()

def plot1DKL(el):
    if el == 'M11':
        grid = pd.read_csv(r"Grid_M11_FP2_100_KL.csv", header = 0)
        x = np.asarray(grid.iloc[:,1])
        y = np.asarray(grid.iloc[:,2])
        plt.plot(x, y, color = 'red')
        plt.title('KL at FP2 vs Q1(M11) when varying Q2 and Q3')
    elif el == 'M12':
        grid = pd.read_csv(r"Grid_M12_FP2_100_KL.csv", header = 0)
        x = np.asarray(grid.iloc[:,1])
        y = np.asarray(grid.iloc[:,2])
        plt.plot(x, y)
        plt.title('KL at FP2 vs Q1(M12) when varying Q2 and Q3')
    elif el == 'M18':
        grid = pd.read_csv(r"Grid_M18_FP2_100_KL.csv", header = 0)
        x = np.asarray(grid.iloc[:,1])
        y = np.asarray(grid.iloc[:,2])
        plt.plot(x, y)
        plt.title('KL at FP2 vs Q1(M18) when varying Q2 and Q3')
    elif el == 'M21':
        grid = pd.read_csv(r"Grid_M21_FP2_100_KL.csv", header = 0)
        x = np.asarray(grid.iloc[:,1])
        y = np.asarray(grid.iloc[:,2])
        plt.plot(x, y, color = 'red')
        plt.title('KL at FP2 vs Q1(M21) when varying Q2 and Q3')
    elif el == 'M33':
        grid = pd.read_csv(r"Grid_M33_FP2_100_KL.csv", header = 0)
        x = np.asarray(grid.iloc[:,1])
        y = np.asarray(grid.iloc[:,2])
        plt.plot(x, y, color = 'red')
        plt.plot([0.5, 0.98], [0, 0], c = 'black', linestyle = "dashed")
        plt.title('KL at FP2 vs Q1(M33) when varying Q2 and Q3')
    plt.xlabel(el)
    plt.ylabel('KL')
    #plt.legend()
    plt.savefig("Q1_el.pdf")
    plt.show()

def plot2DKL():
    grid = pd.read_csv(r"Grid_M11_M33_FP2_KL.csv", header = 0)
    x = np.linspace(0.5, 2.0, 16)    # Hardcoded !!!!!!!!!!!!!!!!!!!!
    y = np.arange(0.5, 0.92, 0.02)    # Hardcoded !!!!!!!!!!!!!!!!!!!!
    KL = grid.iloc[:,3]
    z = np.asarray( KL ).reshape(len(x), len(y)).T
    sm = plt.pcolormesh( x, y, z, norm = colors.SymLogNorm(linthresh=0.5, linscale=0.3, \
        vmin = z.min(), vmax = z.max()) )
    plt.title('KL at FP2 vs Q1(M11) and Q1(M33) when varying Q2 and Q3')
    plt.xlabel('M11')
    plt.ylabel('M33')
    #plt.xlim([10,25])
    #plt.ylim([3,17])
    #plt.clim(0, 20)
    plt.colorbar(sm)
    plt.savefig("M11_M33.pdf")
    plt.show()
