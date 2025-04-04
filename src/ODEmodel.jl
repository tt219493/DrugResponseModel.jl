"""
    In this file we want to estimate parameters of an ODE model describing the
    number of cells in G1 or G2 phase of the cell cycle 
"""

""" Make the transition matrix. """
function ODEjac(p::AbstractVector{T}, nG1 = 8, nG2 = 20)::Matrix{T} where {T <: Real}
    # Original paper: nG1 = 8, nG2 = 20
    # Made to be editable
    
    # Corrections:
    #     p = [a1, a2, a3, a4, b1, b2, b3, b4, g11, g12, g13, g14, g21, g22, g23, g24] = 16 parameters
    #  index = [1,  2,  3,  4,  5,  6,  7,  8,   9,  10,  11,  12,  13,  14,  15,  16]
    # a_i = cells progression rate through ith quarter of G1
    # b_i = cells progression rate through ith quarter of S/G2
    # g1_i = cells rate of death through ith quarter of G1
    # g2_i = cells rate of death through ith quarter of G1

    # values of p come from getODEparams in Hill.jl
    nSp = nG1 + nG2
    A = zeros(nSp, nSp)
    

    A[diagind(A, 0)[1:Int(nG1 / 4)]] .= -(p[1] + p[9])
    A[diagind(A, 0)[Int(nG1 / 4 + 1):Int(nG1 / 2)]] .= -(p[2] + p[10])
    A[diagind(A, 0)[Int(nG1 / 2 + 1):Int(3 * nG1 / 4)]] .= -(p[3] + p[11])
    A[diagind(A, 0)[Int(3 * nG1 / 4 + 1):nG1]] .= -(p[4] + p[12])

    A[diagind(A, 0)[(nG1 + 1):Int(nG1 + nG2 / 4)]] .= -(p[5] + p[13])
    A[diagind(A, 0)[Int(nG1 + nG2 / 4 + 1):Int(nG1 + nG2 / 2)]] .= -(p[6] + p[14])
    A[diagind(A, 0)[Int(nG1 + nG2 / 2 + 1):Int(nG1 + 3 * nG2 / 4)]] .= -(p[7] + p[15])
    A[diagind(A, 0)[Int(nG1 + 3 * nG2 / 4 + 1):nSp]] .= -(p[8] + p[16])

    A[diagind(A, -1)[1:Int(nG1 / 4)]] .= p[1]
    A[diagind(A, -1)[Int(nG1 / 4 + 1):Int(nG1 / 2)]] .= p[2]
    A[diagind(A, -1)[Int(nG1 / 2 + 1):Int(3 * nG1 / 4)]] .= p[3]
    A[diagind(A, -1)[Int(3 * nG1 / 4 + 1):nG1]] .= p[4]

    A[diagind(A, -1)[(nG1 + 1):Int(nG1 + nG2 / 4)]] .= p[5]
    A[diagind(A, -1)[Int(nG1 + nG2 / 4 + 1):Int(nG1 + nG2 / 2)]] .= p[6]
    A[diagind(A, -1)[Int(nG1 + nG2 / 2 + 1):Int(nG1 + 3 * nG2 / 4)]] .= p[7]
    A[diagind(A, -1)[Int(nG1 + 3 * nG2 / 4 + 1):(nSp - 1)]] .= p[8]
    A[1, nSp] = 2 * p[8]
    return A
end

""" Find the starting vector from the steady-state of the control condition. """
function startV(p::AbstractVector{T}, nG1=8, nG2=20)::AbstractVector{T} where {T <: Real}
    @assert all(p .>= 0.0)
    @assert all(p[9:end] .== 0.0) # No cell death in the control
    A = ODEjac(p, nG1, nG2)

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


function vTOg(v::AbstractVector)
    G1 = sum(view(v, 1:nG1))
    G2 = sum(view(v, (nG1 + 1):nSp))
    return G1, G2
end


""" Predicts the model given a set of parametrs. """
function predict(p::AbstractVector, g_0::AbstractVector, t::Union{Real, LinRange}, nG1 = 8, nG2 = 20, g1data = nothing, g2data = nothing)
    @assert length(p) == 16 # we have 4 G1 prog rates, 4 G2 prog rates, 4 G1 death and 4 G2 death rates.

    if length(g_0) == length(p)
        v = startV(g_0, nG1, nG2)
        # v = vcat([0.75/8 for i=1:8], [0.25/20 for i=1:20])
    else
        @assert length(g_0) == nSp
        v = copy(g_0)
    end

    if t isa Real
        A = ODEjac(p, nG1, nG2)
        lmul!(t, A)
        A = LinearAlgebra.exp!(A)

        v = A * v
        G1, G2 = vTOg(v)
    else
        # Some assumptions
        @assert t.start == 0.0
        A = ODEjac(p, nG1, nG2)
        lmul!(t[2], A)
        A = LinearAlgebra.exp!(A)
        u = similar(v)

        if g1data === nothing
            G1 = Vector{eltype(p)}(undef, length(t))
            G2 = Vector{eltype(p)}(undef, length(t))

            for ii = 1:length(t)
                G1[ii], G2[ii] = vTOg(v)

                mul!(u, A, v)
                copyto!(v, u)
            end
        else
            cost = 0.0

            for ii = 1:length(t)
                G1, G2 = vTOg(v)
                cost += norm(G1 - g1data[ii]) + norm(G2 - g2data[ii])

                mul!(u, A, v)
                copyto!(v, u)
            end

            return cost, v
        end
    end

    return G1, G2, v
end
