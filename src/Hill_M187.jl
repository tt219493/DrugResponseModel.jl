"""
This file fits Hill function to the parameters
"""

""" This functions takes in hill parameters for all the concentrations and calculates
DDE parameters, passes them to residual function and based off of these, optimizes the model
and estimates hill parameters. """
function residHill(x::Vector, conc::Vector, g1::Matrix, g2::Matrix, num_parts::Int)
    # n = num_parts
    # x: 2 + 6n + 2 params
    # conc: [8 x 5] (8 concentrations for 5 drugs)
    # g1: [189 x 8 x 5] (189 time points for 96 hours)
    # g2: [189 x 8 x 5]

    # 3 = beginning of max params idx
    # 2 + 2n = end of max params idx

    # max idx + 4n = min idx
    max_end_idx = 2 + 2 * num_parts
    min_idx = 4 * num_parts
    param_end_idx = max_end_idx + min_idx

    res = 0.0
    for i = 3:max_end_idx
        # residual adds 40 * (max - min) ^2 
        res += 40 * (maximum([0, (x[i] - x[i + min_idx])]))^2
    end
    # println(res)
    # params: [4n x 8 x 1] (4n params, 8 concs)
    params = getODEparams(x[1:param_end_idx], conc, num_parts)
    
    nG1 = trunc(Int, x[param_end_idx + 1])
    nG2 = trunc(Int, x[param_end_idx + 2])
    
    #nG1 = 2
    #nG2 = 5
    
    t = LinRange(0.0, 0.5 * size(g1, 1), size(g1, 1)) # 0.0 to 94.5; 189 steps

    # Solve each concentration separately
    for ii = 1:length(conc)
        res += predict(params[:, ii, 1], params[:, 1, 1], t, num_parts, nG1, nG2, g1[:, ii], g2[:, ii])[1] 
        #println("Res: ", res)
    end
    return res
end


""" Generic setup for optimization. """
function optimize_helper(f, low::Vector, high::Vector, maxstep::Int)
    results_ode = bboptimize(
        f;
        SearchRange = collect(zip(low, high)),
        NumDimensions = length(low),
        TraceMode = :verbose,
        TraceInterval = 100,
        MaxSteps = maxstep,
        Method = :adaptive_de_rand_1_bin_radiuslimited
    )

    return best_fitness(results_ode), best_candidate(results_ode)
end


""" Hill optimization function. """
function optimize_hill(conc::Vector, g1::Matrix, g2::Matrix; maxstep = 200000, num_parts = 4, 
    g1_lower = 2, g1_upper = 2, g2_lower = 5, g2_upper = 5)
    # num_parts: hyperparameter to choose number of parts to use in Jacobian
        # default used in paper: 4
        # default: 28 params
        # nG1 default: 2
        # nG2 default: 5

    f(x) = residHill(x, conc, g1, g2, num_parts)

    # add nG1 and nG2 to fit for best number of sub-parts

    # [EC50, k, 
    # max_a1,  ..., max_an, 
    # max_b1,  ..., max_bn, 
    # max_g11, ..., max_g1n,
    # max_g21, ..., max_g2n, 
    # min_a1,  ..., min_an,
    # min_b1,  ..., min_b2,
    # nG1, nG2]

    # 2 + 6n + 2 parameters
    # nG1 test range: 1 to 4
    # nG2 test range: 1 to 10
    n = num_parts
    low =  [minimum(conc);   1e-9 * ones(1 + 6*n) ; g1_lower; g2_lower]
    high = [2*maximum(conc); 50.0; 4.0 * ones(6*n); g1_upper; g2_upper]

    return optimize_helper(f, low, high, maxstep)
end


function getODEparams(p, conc, num_parts)
    # Default:
        # length of p = 26
        # nMax = 1
        # nMax = Int((length(p) - 8) / 18)

        # effects: (16, 8, nMax) = (16, 8, 1)
        # effects = zeros(eltype(p), 16, length(conc[:, 1]), nMax)

        # k = 1
        # sizep = 18 # the size of independent parameters, meaning except for control.

        # [EC50, k, 16 max params, unused in function: 3 params for num of phases]

        # j = sizep + 1 # the starting index of "control parameters", according to the number of drugs being fitted at once.
        # [16 min params]
    
    nMax = 1
    num_params = 4*num_parts
    num_prog = 2*num_parts # number of progression params (a_n, b_n)

    effects = zeros(eltype(p), num_params, length(conc[:, 1]), nMax)    

    k = 1
    j = 2 + num_params # number of independent params (j + 1 = starting index of control params)


    # Scaled drug effect
    for i = 1:nMax
        xx = 1.0 ./ (1.0 .+ (p[k] ./ conc[:, i]) .^ p[k + 1]) 
        # Hill Function w/o E_min and E_max
        # p[k]   = p[1] = EC50
        # p[k+1] = p[2] = k

        # [EC50, left, right, steepness]

        # p[j + ...] = E_min
        # p[k + ...] = E_max

        # a_1 ... a_n, b_1, ... b_n
        for param = 1:num_prog
            effects[param, :, i] = p[j + param] .+ (p[k + 1 + param] - p[j + param]) .* xx
        end

        # g_11 ... g_1n, g_21 ... g_2n
        for param = (num_prog+1):num_params
            # no deaths in control (E_min = 0)
            effects[param, :, i] = p[k + 1 + param] .* xx
        end

        k += j
    end
    return effects
end

hill_func(p, conc) = p[4] .+ (p[3] - p[4]) ./ (1.0 .+ (p[1] ./ conc) .^ p[2])


function costingss(pp, total, concs)
    cost = 0
    k = 1
    for i = 1:5
        cost += sum((hill_func(pp[k:(k + 3)], concs[:, i]) .- total[:, i]) .^ 2)
        k += 4
    end
    return cost
end
