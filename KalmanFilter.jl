# This module export kalman filter structures and function.

# just for simple Linear system.

#=
The normal workflw is like the:

```julia

# initialize all arguments
x0, P0 = x0, P0 
record_list = [...]
movement = Movement{T}(F, Q, B, u)
Observation = Observation{T}(H, R)

# predict 
x_update, P_update = x0, P0

for record in record_list
    x_predict, P_predict = movement(...)
    ...
    ...storage the argument you need, maybe push!(...)
    ...
    x_update, P_update, K = observation(...)
end

now you can use the data product by the cycling
````
=#
module KalmanFilter

using  LinearAlgebra

export Kalmanfilter, ClassicKalmanfilter, Movement,  Observation, check_kalman, Sigma, Noise

const LAB = LinearAlgebra
"""
abstract struct Kalmanfilter include 2 structures.

    Movement : storage the Linear system movement arguments
    Observation: storage the transform from movement system to Observation and noise of Observation.
"""
abstract type Kalmanfilter end
abstract type ClassicKalmanfilter <: Kalmanfilter end
abstract type UnscentedKalmanfilter <: Kalmanfilter end


"""
Movement show how the system moves according to matrix (Linear System)
    F : x_k = F * x_k-1 + B * u_k + w_k
    Q : covariance of w_k (noise strength and interact)
    B : count the influence of Input u_k to the movement
    u : Input, maybe a control variable
"""
mutable struct Movement{T <: Real} <: ClassicKalmanfilter
    F::Matrix{T}
    Q::Matrix{T}
    B::Matrix{T}
    u::Vector{T}
    Movement(F, Q, B, u) = (size(F) == size(Q) && size(B) == (size(F, 1), length(u))) ? new{eltype(u)}(F, Q, B, u) : error("Dimension does't matc")
end
Movement(F::Matrix{T}, Q::Matrix{T}) where T <: Real = Movement(F, Q, zeros(T, size(Q)...), zeros(T, size(Q, 1)))

"""
Observation tells how the state of movement tranfer to Observation.
    H : Observation = H * x
    R : covariance of observing noise.
"""
mutable struct Observation{T <: Real} <: ClassicKalmanfilter
    H::Matrix{T}
    R::Matrix{T}
    Observation(H, R) = (size(H, 1) == size(R, 1)) ? new{eltype(H)}(H, R) : error("Dimension does't match!")
end

"""
check_kalman(x0::Vector{T}, P0::Matrix{T}, record::Vector{T}, intr::Interior{T}, move::Movement{T}, obsv::Observation{T})
check out the argument of kalman filter.
"""
function check_kalman end

function check_kalman(x0::Vector{T}, P0::Matrix{T}, record::Vector{T},  move::Movement{T}, obsv::Observation{T}) where T <: Real
    move_dims = length(x0)
    obsv_dims = length(record)

    size(P0) == (move_dims, move_dims) || println("DimensionMissmatch : initial covariance P0 should have $(move_dims) rows and $(move_dims) columns")
    size(move.F, 1) == move_dims || println("DimensionMissmatch : Movement struct should have $(move_dims) rows")
    size(obsv.R, 1) == obsv_dims || println("DimensionMissmatch : Observation struct should have $(obsv_dims) rows")
    size(obsv.H, 1) == move_dims || println("DimensionMissmatch : Observation.H should have $(move_dims) columns")
end


"""
x_predict, P_predict = (move::Movement{T})(x_update::Vector{T}, P_update::Matrix{T})
return the prediction of x and P when we know the movement infornmation.
"""

function (move::Movement{T})(x_update::Vector{T}, P_update::Matrix{T}) where T <: Real
    x_predict = move.F * x_update + move.B * move.u
    P_predict = move.F * P_update * transpose(move.F) + move.Q
    return (x_predict, P_predict)
end


function eye(dim::Int, T::DataType = Float64)
    matrix = zeros(T, dim, dim)
    for j in 1:dim
        @inbounds matrix[j, j] = one(T)
    end
    return matrix
end

"""
x_update, P_update, K = (obsv::Observation{T})(x_predict::Vector{T}, P_predict::Matrix{T}, record::Vector{T})
return the update for x , P and K.
"""

function (obsv::Observation{T})(x_predict::Vector{T}, P_predict::Matrix{T}, record::Vector{T}) where T <: Real
    δ = record - obsv.H * x_predict
    S = obsv.H * P_predict * transpose(obsv.H) + obsv.R
    K = P_predict * transpose(obsv.H) / S
    
    
    I = eye(length(x_predict), T)
    x_update = x_predict + K * δ
    P_update = (I - K * obsv.H) * P_predict
    
    return x_update, P_update, K
end

"""
mutable struct Sigma{T <: Real} <: UnscentedKalmanfilter
    storage arguments α, β, κ
    default β = 2, κ = 3 - N
"""
mutable struct Sigma{T <: Real} <: UnscentedKalmanfilter
    α::T
    β::T
    κ::T
end
Sigma(α::T, β::T, κ::T) where T <: Real = Sigma{typeof(α)}(α, β, κ)
Sigma(α::T, dims::Int) where T<: Real = Sigma(α, 2.0, 3.0 - dims)

"""
mutable struct Noise{T <: Real} <: UnscentedKalmanfilter
    define movement noise and observation noise
"""
mutable struct Noise{T <: Real} <: UnscentedKalmanfilter
    Q::Matrix{T}
    R::Matrix{T}
    Noise(Q, R) = new{eltype(Q)}(Q, R) 
end

"""
function (sigma::Sigma{T})(x::Vector{T}, P::Matrix{T}) where T <: Real
    sigmaPoints, wm0, wc0, wm, wc = sigma(x, P)
    generate sigma points and its weights.
"""
function (sigma::Sigma{T})(x::Vector{T}, P::Matrix{T}) where T <: Real
    α, β, κ = sigma.α, sigma.β, sigma.κ
    N = length(x)
    Lmatrix = LAB.cholesky(P |> LAB.Hermitian).L
    λ = α^2 * (N + κ) - N

    wm0 = λ / (N + λ)
    wm = 0.5 / (N + λ)
    wc0 = λ / (N + λ) + 1 - α^2 + β
    wc = 0.5 / (N + λ)

    sigmaPoints = Matrix{T}(undef, N, 2*N)

    @inbounds for i in 1:N
        sigmaPoints[:, 2*i-1] = x + α * √(N + κ) * Lmatrix[:, i]
        sigmaPoints[:, 2*i] = x - α * √(N + κ) * Lmatrix[:, i]
    end

    return sigmaPoints, wm0, wc0, wm, wc

end

function (noise::Noise{T})(x::Vector{T}, P::Matrix{T}, zrecord::Vector{T}, sigma::Sigma{T}, mtransform, otransform) where T <: Real
    N = length(x)
    Q, R = noise.Q, noise.R

    sigmaPoints, wm0, wc0, wm, wc = sigma(x, P)
    y0 = mtransform(x)
    ylist = mapslices(mtransform, sigmaPoints, dims = 1)

    # movement forecast, count ymean, pymean as forecast 
    ymean = wm0 * y0 + ylist*fill(wm, 2*N)
    
    δylist = ylist .- ymean
    δy = y0 - ymean 
    pymean = wc0 * δy * transpose(δy) + wc * δylist * transpose(δylist) + Q

    # updata for observation
    z0 = otransform(y0)
    zlist = mapslices(otransform, ylist, dims = 1)

    zmean = wm0 * z0 + zlist*fill(wm, 2*N)

    δzlist = zlist .- zmean
    δz = z0 - zmean 
    pzmean = wc0 * δz * transpose(δz) + wc * δzlist * transpose(δzlist) + R
    pxzmean = wc0 * δy * transpose(δz) + wc * δylist * transpose(δzlist) 
    K = pxzmean / pzmean

    δx = zrecord - zmean

    x_update = ymean + K*δx
    P_update = pymean - K*pzmean*transpose(K)

    return x_update, P_update, K

end

end

