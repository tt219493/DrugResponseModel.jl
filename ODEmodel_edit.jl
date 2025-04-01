"""
    In this file we want to estimate parameters of an ODE model describing the
    number of cells in G1 or G2 phase of the cell cycle 
"""

""" Make the transition matrix. """
function ODEjac(p::AbstractVector{T}, num_parts::Int, nG1::Int, nG2::Int)::Matrix{T} where {T <: Real}
    
    # a_i = cells progression rate through ith quarter of G1
    # b_i = cells progression rate through ith quarter of S/G2
    # g1_i = cells rate of death through ith quarter of G1
    # g2_i = cells rate of death through ith quarter of G1

    nG1_tot = num_parts * nG1
    nG2_tot = num_parts * nG2
    nSp =  nG1_tot + nG2_tot
    A = zeros(nSp, nSp)
    
    # -(progression + death) rates for itself
    for i = 0:(num_parts-1)
        A[diagind(A, 0)[(1 + nG1 * i) : (nG1 * (i+1))]] .= -(p[i+1] + p[i+1 + 2*num_parts])
        A[diagind(A, 0)[(1 + nG1_tot + nG2 * i):(nG1_tot + nG2 * (i+1))]] .= -(p[num_parts+i+1] + p[num_parts+i+1 + 2*num_parts])
        A[diagind(A, -1)[(1 + nG1 * i) : (nG1 * (i+1))]] .= p[i+1]
        A[diagind(A, -1)[(1 + nG1_tot + nG2 * i):(nG1_tot + nG2 * (i+1))]] .= p[num_parts+i+1]
    end
    
    A[1, nSp] = 2 * p[2*num_parts]

    return A
end

""" Find the starting vector from the steady-state of the control condition. """
function startV(p::AbstractVector{T}, num_parts::Int, nG1::Int, nG2::Int)::AbstractVector{T} where {T <: Real}
    death_idx = (length(p) / 2) + 1
    @assert all(p .>= 0.0)
    @assert all(p[death_idx:end] .== 0.0) # No cell death in the control

    A = ODEjac(p, num_parts, nG1, nG2)

    vals, vecs = eigen(A)

    a = real.(vals) .> 0.0
    select = imag.(vals) .== 0.0
    selectt = a .* select
    @assert sum(selectt) == 1
    vecs = vec(vecs[:, selectt])
    @assert all(isreal.(vals[selectt]))
    @assert all(isreal.(vecs))

    return vecs / sum(vecs)
end


function vTOg(v::AbstractVector, num_parts::Int, nG1::Int, nG2::Int)
    nG1_tot = num_parts * nG1 
    nG2_tot = num_parts * nG2
    nSp = nG1_tot + nG2_tot

    G1 = sum(view(v, 1:nG1_tot))
    G2 = sum(view(v, (nG1_tot + 1):nSp))
    return G1, G2
end


""" Predicts the model given a set of parametrs. """
function predict(p::AbstractVector, g_0::AbstractVector, t::Union{Real, LinRange}, num_parts::Int, nG1::Int, nG2::Int, g1data = nothing, g2data = nothing)
    # g_0 = params for 0 concentration (control)
    nSp = num_parts * (nG1 + nG2)
    @assert length(p) == 4*num_parts # we have n G1 prog rates, n G2 prog rates, n G1 death and n G2 death rates.

    if length(g_0) == length(p)
        v = startV(g_0, num_parts, nG1, nG2)
    else
        @assert length(g_0) == nSp
        v = copy(g_0)
    end

    #lmul! = left-multiplication
    if t isa Real
        A = ODEjac(p, num_parts, nG1, nG2)
        lmul!(t, A)
        A = LinearAlgebra.exp!(A)

        v = A * v
        G1, G2 = vTOg(v, num_parts, nG1, nG2)
    else
        # Some assumptions
        @assert t.start == 0.0
        A = ODEjac(p, num_parts, nG1, nG2)
        lmul!(t[2], A)
        A = LinearAlgebra.exp!(A)
        u = similar(v)

        if g1data === nothing
            G1 = Vector{eltype(p)}(undef, length(t))
            G2 = Vector{eltype(p)}(undef, length(t))

            for ii = 1:length(t)
                G1[ii], G2[ii] = vTOg(v, num_parts, nG1, nG2)

                mul!(u, A, v)
                copyto!(v, u)
            end
        else
            cost = 0.0

            for ii = 1:length(t)
                G1, G2 = vTOg(v, num_parts, nG1, nG2)
                cost += norm(G1 - g1data[ii]) + norm(G2 - g2data[ii])

                mul!(u, A, v)
                copyto!(v, u)
            end

            return cost, v
        end
    end

    return G1, G2, v
end
