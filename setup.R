library(hydromad)

data(Cotter)
x <- Cotter[1:1000]

## IHACRES CWI model with exponential unit hydrograph
## an unfitted model, with ranges of possible parameter values
modspec <- hydromad(x, sma = "cwi", routing = "expuh",
                 tau_s = c(2,100), v_s = c(0,1))

## now try to fit it
modfit <- fitByOptim(modx)

## Define the quantity of interest that will be used in HEAT
## In this case NSE*
objective <- function (Q, X, ...)
{
  nseStat(coredata(Q), coredata(X), ...)/(2 - nseStat(coredata(Q), coredata(X), ...))
}

## Create the python wrapper for HEAT
source("pyWrapperHeat.R")
pyWrapperHeat(modspec,modfit,objective,"Cotter_cwi_expuh")


