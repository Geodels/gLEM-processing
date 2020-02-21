import argparse
import gLEM as sim


# Parsing command line arguments
parser = argparse.ArgumentParser(description='This is a simple entry to run eSCAPE model.',add_help=True)
parser.add_argument('-i','--input', help='Input file name (YAML file)',required=True)
parser.add_argument('-v','--verbose',help='True/false option for verbose',required=False,action="store_true",default=False)
parser.add_argument('-l','--log',help='True/false option for PETSC log', required=False,action="store_true",default=False)

args = parser.parse_args()

# Reading input file
model = sim.LandscapeEvolutionModel(args.input,args.verbose,args.log)

bestfit = True

# Iterative accuracy check based on paleomap fitting
while(model.runNb <= model.paleostep):

    # Running forward model
    model.runProcesses()

    # Differences between simulation and reference paleomaps
    model.comparePaleomaps(verbose=True)
    model.runNb += 1

    # Advect elevation differences backward
    if model.runNb > 1:
        if model.improvement[-1].values[0][1] < model.cvglimit: # and model.accuracy[-1].values[0][-2]<0.1:
            if model.improvement[-1].values[0][1] > 0.:
                model.backwardAdvection(verbose=False)
            elif abs(model.improvement[-1].values[0][1]) < model.cvglimit:
                model.backwardAdvection(verbose=False)
            else:
                bestfit = False
            break
        if model.runNb > model.paleostep:
            model.backwardAdvection(verbose=False)
            break
    model.backwardAdvection(verbose=False)

    # Reinitialise forward model
    model.reInitialise()


if bestfit:
     
    # Reinitialise forward model
    model.reInitialise()

    # Running forward model
    model.runProcesses()

# Paleomap forcing
if model.runNb <= model.paleostep:
    model.forcePaleomap()
    model.matchPaleo()

# Differences between simulation and reference paleomaps
model.comparePaleomaps(verbose=False,disk=True)


# Cleaning model
model.destroy()
