from pce_hydromad import *
from Cotter_cwi_expuh import *

model=Cotter_cwi_expuh()
rv_trans = define_random_variable_transformation_hydromad(model)

# Regression
# Draw sample if pts.txt and vals.txt don't exist
construct_data(model,num_pts=20000)
pce = build_pce_regression( 'pts.txt', 'vals.txt', rv_trans )
   
# Sparse grid
#pce = build_pce_sparse_grid(rv_trans, max_num_points=20000, model)

# Output
plot_sensitivities( pce )
x = get_sensitivities( pce ) #me,te,ie
numpy.savetxt("sensitivities.csv",numpy.transpose(x),delimiter=",")
numpy.savetxt("interactions.csv",get_interaction_values(pce),delimiter=",",fmt='"%s"')
error=get_pce_error(pce,model,num_pts=1000)
with open("error_pce.txt","wb") as f: f.write("%f" % error)
error_calib=get_pce_error_calib(pce,model,'pts.txt','vals.txt')
with open("error_pce_calib.txt","wb") as f: f.write("%f" % error_calib)
