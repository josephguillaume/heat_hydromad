import numpy
from scipy.misc import comb as nchoosek
from polynomial_chaos_cpp import PolynomialChaosExpansion, LegendrePolynomial1D,\
    initialise_homogeneous_pce
from model_cpp import RandomVariableTransformation
from indexing_cpp import PolyIndexVector

from utilities.generalised_sparse_grid_tools import get_quadrature_rule_1d,\
    convert_homogeneous_sparse_grid_to_pce, build_generalised_sparse_grid,\
    HierarchicalSurplusDimensionRefinementManager


def get_sensitivities( pce, qoi = 0 ):
    main_effects = numpy.zeros( ( pce.dimension() ), numpy.double )
    total_effects = numpy.zeros( ( pce.dimension() ), numpy.double )
    interaction_effects = numpy.zeros( ( pce.dimension()+1 ), numpy.double )
    var = pce.variance()
    indices = PolyIndexVector()
    pce.get_basis_indices( indices )
    coeff = pce.get_coefficients()[:,qoi]
    for index in indices:
        # calculate contibution to variance of the index
        var_contribution = \
            coeff[index.get_array_index()]**2*pce.l2_norm( index )
        # get number of dimensions involved in interaction, also known 
        # as order
        order = index.num_effect_dims()
        # update interaction effects
        interaction_effects[order] += var_contribution
        # update main effects
        if ( order == 1 ):
            dim = index.dimension_from_array_index( 0 )
            main_effects[dim] += var_contribution
        # update total effects
        for i in xrange( order ):
            dim = index.dimension_from_array_index( i )
            total_effects[dim] += var_contribution
    main_effects /= var
    total_effects /= var
    interaction_effects /= var
    return main_effects, total_effects, interaction_effects[1:]

def get_interactions( pce, qoi = 0 ):
    interactions = dict()
    indices = PolyIndexVector()
    pce.get_basis_indices( indices )
    coeff = pce.get_coefficients()[:,qoi]
    for index in indices:
        var_contribution = \
            coeff[index.get_array_index()]**2*pce.l2_norm( index )
        key = ()
        for i in xrange( index.num_effect_dims() ):
            dim = index.dimension_from_array_index( i )
            key += (dim,)
        if len( key ) > 0 and interactions.has_key( key ):
            interactions[key] += var_contribution
        elif len( key ) > 0:
            interactions[key] = var_contribution

    # convert interactions to a list for easy sorting
    interaction_terms = []
    interaction_values = []
    for key in interactions.keys():
        interaction_terms.append( numpy.array( key, numpy.int32 ) )
        interaction_values.append( interactions[key] )

    # sort interactions in descending order
    I = numpy.argsort( interaction_values )
    values_tmp = numpy.empty( ( len( I ) ), numpy.double )
    terms_tmp = []
    for i in xrange( len( I ) ):
        values_tmp[i] = interaction_values[I[i]]
        terms_tmp.append( interaction_terms[I[i]] )
    return values_tmp[::-1], terms_tmp[::-1]

def get_interaction_values(pce):
    interaction_values, interaction_terms = get_interactions( pce )
    interaction_values /= interaction_values.sum()
    labels = []
    for i in xrange( interaction_values.shape[0] ):
        l = '($'
        for j in xrange( len( interaction_terms[i] )-1 ):
            l += 'z_{%d},' %(interaction_terms[i][j]+1)
        l+= 'z_{%d}$)' %(interaction_terms[i][-1]+1)
        labels.append( l )
    #interaction_values = interaction_values[:i+1]
    assert interaction_values.shape[0] == len ( labels )
    return(numpy.transpose(numpy.asarray([labels,interaction_values])))

def construct_data(model,num_pts,seed=None):
    num_dims = model.num_dims
    rv_trans = define_random_variable_transformation_hydromad(model)
    rng = numpy.random.RandomState( seed )
    pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    pts = rv_trans.map_from_canonical_distributions( pts )
    vals = model.evaluate_set( pts )                  
    numpy.savetxt( 'pts.txt', pts, delimiter = ',' )
    numpy.savetxt( 'vals.txt', vals, delimiter = ',' )

def build_pce_regression( pts_filename, vals_filename, rv_trans ):
    # Must be a ( num_dims x num_pts ) matrix
    pts = numpy.loadtxt( pts_filename, delimiter = ',' )
    # must be a ( num_pts x 1 ) vector
    vals = numpy.loadtxt( vals_filename, delimiter = ',' )
    vals = vals.reshape( vals.shape[0], 1 )

    num_dims, num_pts = pts.shape 

    # find degree of PCE
    degree = 2
    while ( True ):
        num_basis_terms = nchoosek( degree + num_dims, num_dims )
        if ( num_basis_terms > num_pts ):
            break
        degree += 1
    degree -= 1

    # define the parameters of the PCE
    pce = PolynomialChaosExpansion()
    pce.set_random_variable_transformation( rv_trans )
    pce.define_isotropic_expansion( degree, 1. )

    # form matrices needed for normal equations 
    V, build_vals = pce.build_linear_system( pts, vals, 
                                             False )
    assert V.shape[1] <= V.shape[0]

    # Solve least squares to find PCE coefficients    
    coeff = numpy.linalg.solve( numpy.dot( V.T, V ), 
                                numpy.dot( V.T, build_vals ) )
    pce.set_coefficients( coeff.reshape( coeff.shape[0], 1 ) )

    return pce

def build_pce_sparse_grid(rv_trans, max_num_points , model):
    quad_type = 'clenshaw-curtis'
    quadrature_rule_1d, orthog_poly_1d = get_quadrature_rule_1d( quad_type )
    rm = None
    max_level = numpy.iinfo( numpy.int32 ).max
    max_level_1d = numpy.iinfo( numpy.int32 ).max
    sg = build_generalised_sparse_grid( quadrature_rule_1d, orthog_poly_1d,
                                        rv_trans, model, max_num_points,
                                        tpqr = None, rm = rm, tolerance = 0., 
                                        max_level = max_level, 
                                        max_level_1d = max_level_1d,
                                        verbosity = 0,
                                        test_pts = None, 
                                        test_vals = None, 
                                        breaks = None )[0]

    pce = PolynomialChaosExpansion()
    pce.set_random_variable_transformation( rv_trans )
    sg.convert_to_polynomial_chaos_expansion( pce, 0 )
    
    return pce

from utilities.visualisation import *
def plot_sensitivities( pce ):
    me, te, ie = get_sensitivities( pce )
    interaction_values, interaction_terms = get_interactions( pce )
    
    show = False
    fignum = 1
    filename = 'individual-interactions.png'
    plot_interaction_values( interaction_values, interaction_terms, 
                             title = 'Sobol indices', truncation_pct = 0.95, 
                             filename = filename, show = show,
                             fignum = fignum, rv = r'\xi' )
    fignum += 1
    filename = 'dimension-interactions.png'
    plot_interaction_effects( ie, title = 'Dimension-wise joint effects', 
                              truncation_pct = 0.95, filename = filename, 
                              show = show, fignum = fignum )
    fignum += 1
    filename = 'main-effects.png'
    plot_main_effects( me, truncation_pct = 0.95, 
                       title = 'Main effect sensitivity indices', 
                       filename = filename, show = show, fignum = fignum  )
    fignum += 1
    filename = 'total-effects.png'
    plot_total_effects( te, truncation_pct = 0.95, 
                        title = 'Total effect sensitivity indices', 
                        filename = filename, show = show, fignum = fignum,
                        rv = r'\xi'   )
    fignum += 1

    pylab.show()

def define_random_variable_transformation_hydromad(model):
    num_dims=model.num_dims
    rv_trans = RandomVariableTransformation()
    dist_types = ['uniform']*num_dims
    means = numpy.zeros( ( num_dims ), numpy.double )    # dummy for uniform
    std_devs = numpy.zeros( ( num_dims ), numpy.double ) # dummy for uniform
    ranges=model.domain
    #ranges = numpy.array( [0.08,0.12,0.03,0.04,0.08,0.12,0.8,1.2,
    #                       0.45,0.55,-0.05,0.05], numpy.double )
    rv_trans.set_random_variables( dist_types, ranges, means, std_devs )
    return rv_trans


def get_pce_error(pce,model,num_pts=10000,seed=None):
    num_dims=model.nominal_z.shape[0]
    rv_trans = define_random_variable_transformation_hydromad(model)
    rng = numpy.random.RandomState( seed )
    test_pts = rng.uniform( -1., 1., ( num_dims, num_pts ) )
    test_pts = rv_trans.map_from_canonical_distributions( test_pts )
    print "evaluating %d points" % num_pts
    test_vals = numpy.asarray(model.evaluate_set( test_pts ))                  
    return(numpy.linalg.norm( test_vals.squeeze() - pce.evaluate_set( test_pts ).squeeze() )  / numpy.sqrt( test_vals.shape[0] ))

## Adapted from Mun-Ju, 13.6.2014
def get_pce_error_calib(pce,model, pts_filename, vals_filename):
    pts = numpy.loadtxt( pts_filename, delimiter = ',' )
    vals = numpy.loadtxt( vals_filename, delimiter = ',' )
    vals = vals.reshape( vals.shape[0], 1 )
    rv_trans = define_random_variable_transformation_hydromad(model)
    return(numpy.linalg.norm( vals.squeeze() - pce.evaluate_set( pts ).squeeze() )  / numpy.sqrt( test_vals.shape[0] ))


########################################################
## Calculate bootstrap estimates

def get_tsi( pce, qoi = 0 ):
    total_effects = numpy.zeros( ( pce.dimension() ), numpy.double )
    var = pce.variance()
    indices = PolyIndexVector()
    pce.get_basis_indices( indices )
    coeff = pce.get_coefficients()[:,qoi]
    for index in indices:
        # calculate contibution to variance of the index
        var_contribution = \
            coeff[index.get_array_index()]**2*pce.l2_norm( index )
        # get number of dimensions involved in interaction, also known
        # as order
        order = index.num_effect_dims()
        # update total effects
        for i in xrange( order ):
            dim = index.dimension_from_array_index( i )
            total_effects[dim] += var_contribution
    total_effects /= var
    return total_effects

import scikits.bootstrap as bootstrap
def bootstrap_pce_regression(pts_filename, vals_filename,rv_trans,alpha=0.05,n_samples=3000):
    # Must be a ( num_dims x num_pts ) matrix
    pts = numpy.loadtxt( pts_filename, delimiter = ',' )
    # must be a ( num_pts x 1 ) vector
    vals = numpy.loadtxt( vals_filename, delimiter = ',' )
    vals = vals.reshape( vals.shape[0], 1 )
    #data=numpy.hstack((pts.transpose(),vals))
    
    def bootstrappable_pce_regression(pts,vals):
        ## bootstrap gives this function a tuple of arrays of shape (N,...)
        ## but PCE expects pts to be of shape (...,N), so we transpose
        pts=pts.transpose()
        num_dims, num_pts = pts.shape
        #num_dims-= 1
        #pts = data[:,range(num_dims)]
        #vals = data[:,num_dims]

         # find degree of PCE
        degree = 2
        while ( True ):
            num_basis_terms = nchoosek( degree + num_dims, num_dims )
            if ( num_basis_terms > num_pts ):
                break
            degree += 1
        degree -= 1

        # define the parameters of the PCE
        pce = PolynomialChaosExpansion()
        pce.set_random_variable_transformation( rv_trans )
        pce.define_isotropic_expansion( degree, 1. )

        # form matrices needed for normal equations
        V, build_vals = pce.build_linear_system( pts, vals,
                                                 False )
        assert V.shape[1] <= V.shape[0]

        # Solve least squares to find PCE coefficients
        coeff = numpy.linalg.solve( numpy.dot( V.T, V ),
                                    numpy.dot( V.T, build_vals ) )
        pce.set_coefficients( coeff.reshape( coeff.shape[0], 1 ) )
        return get_tsi(pce,qoi=0)
        
    TSIs=bootstrap.ci((pts.transpose(),vals),bootstrappable_pce_regression,alpha=alpha,n_samples=n_samples,multi=True)

    return TSIs

if __name__ == "__main__":
    pass
