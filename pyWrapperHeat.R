library(hydromad)

## Allow hydromad model to be used as python model in HEAT
pyWrapperHeat<-function(modspec,modfit,objective,name){

  save(modspec,modfit,objective,file=sprintf("%s.Rdata",name))

  if(length(objective)>1){
    pywrap_heat=sprintf('import numpy
from rpy2.robjects import r
from model_cpp import Model
class %s(Model):
        def __init__(self):
                Model.__init__( self ) # must call c++ object wrapper init function
                r("library(hydromad)")
                r("load(\'%s\')")
                self.domain=numpy.hstack(r("getFreeParsRanges(modspec)"))
                self.nominal_z=numpy.hstack(r("coef(modfit)[names(getFreeParsRanges(modspec))]"))
                self.num_qoi=%d
                self.num_dims=%d

        def evaluate(self,z):
                return numpy.array(r(\'\'\'sapply(objective,function(obj) objFunVal(update(modspec,
%s
),objective=obj))\'\'\' %% tuple(z))[0])

if __name__=="__main__":
        m=%s()
        print m.evaluate(m.nominal_z)
        print m.evaluate_set(numpy.vstack((m.nominal_z,m.nominal_z)).T)
',
      name,
      sprintf("%s.Rdata",name),
      length(objective),
      length(getFreeParsRanges(modspec)),
      paste(sprintf("%s=%%f",names(getFreeParsRanges(modspec))),collapse=",\n"),
      name
      )
  } else if (length(objective)==1){
    pywrap_heat=sprintf('import numpy
from rpy2.robjects import r
from model_cpp import Model
class %s(Model):
        def __init__(self):
                Model.__init__( self ) # must call c++ object wrapper init function
                r("library(hydromad)")
                r("load(\'%s\')")
                self.domain=numpy.hstack(r("getFreeParsRanges(modspec)"))
                self.nominal_z=numpy.hstack(r("coef(modfit)[names(getFreeParsRanges(modspec))]"))
                self.num_qoi=1
                self.num_dims=%d

        def evaluate(self,z):
                return numpy.array([r(\'\'\'objFunVal(update(modspec,
%s
),objective=objective)\'\'\' %% tuple(z))[0]])

if __name__=="__main__":
        m=%s()
        print m.evaluate(m.nominal_z)
        print m.evaluate_set(numpy.vstack((m.nominal_z,m.nominal_z)).T)
',
      name,
      sprintf("%s.Rdata",name),
      length(getFreeParsRanges(modspec)),
      paste(sprintf("%s=%%f",names(getFreeParsRanges(modspec))),collapse=",\n"),
      name
      )
  }

  cat(pywrap_heat,file=sprintf("%s.py",name))

  cat(sprintf("Wrote %s.Rdata and %s.py\n",name,name))

}
