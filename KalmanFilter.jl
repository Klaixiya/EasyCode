# This module export kalman filter structures and function.

# just for simple Linear system.

#=
The normal workflw is like the:

```julia

# initialize all arguments
x0, P0 = x0, P0 
record_list = [...]
intr = Interior{T}(row, collumn)
movement = Movement{T}(F, Q, B, u)
Observation = Observation{T}(H, R)

# predict 
x_update, P_update = x0, P0

for record in record_list
    x_predict, P_predict = predict_kalman(...)
    ...
    ...storage the argument you need, maybe push!(...)
    ...
    x_update, P_update = update_kalman!(...)
end

now you can use the data product by the cycling
````
=#
module KalmanFilter
 
export Kalmanfilter,  Interior, Movement,  Observation,check_kalman,  predict_kalman, update_kalman!

"""
abstract struct Kalmanfilter include 3 structures.

    Interior : storage the filter Interior arguments
    Movement : storage the Linear system movement arguments
    Observation: storage the transform from movement system to Observation and noise of Observation.
"""
abstract type Kalmanfilter end
abstract type ClassicKalmanfilter <: Kalmanfilter end
abstract type UnscentedKalmanfilter <: Kalmanfilter end

"""
Interior explain how the estimation is done:
    δ : Observation - predict_of_movement
    S : pre-fit residual
    K : Optimal Kalman gain
"""
mutable struct Interior{T <: Real} <: ClassicKalmanfilter
    δ::Vector{T}
    S::Matrix{T}
    K::Matrix{T}
end
Interior(row::Int, column::Int, T::DataType = Float64) = Interior{T}(zeros(T, column), zeros(T, column, column), zeros(T, row, column))

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
    Movement(F, Q, B, u) = (size(F) == size(Q) && size(B) == (size(F)[1], length(u))) ? new{eltype(u)}(F, Q, B, u) : error("Dimension does't matc")
end
Movement(F::Matrix{T}, Q::Matrix{T}) where T <: Real = Movement(F, Q, zeros(T, size(Q)...), zeros(T, size(Q)[1]))

"""
Observation tells how the state of movement tranfer to Observation.
    H : Observation = H * x
    R : covariance of observing noise.
"""
mutable struct Observation{T <: Real} <: ClassicKalmanfilter
    H::Matrix{T}
    R::Matrix{T}
    Observation(H, R) = (size(H)[1] == size(R)[1]) ? new{eltype(H)}(H, R) : error("Dimension does't match!")
end

"""
check_kalman(x0::Vector{T}, P0::Matrix{T}, record::Vector{T}, intr::Interior{T}, move::Movement{T}, obsv::Observation{T})
check out the argument of kalman filter.
"""
function check_kalman end

function check_kalman(x0::Vector{T}, P0::Matrix{T}, record::Vector{T}, intr::Interior{T}, move::Movement{T}, obsv::Observation{T}) where T <: Real
    move_dims = length(x0)
    obsv_dims = length(record)

    size(P0) == (move_dims, move_dims) || println("DimensionMissmatch : initial covariance P0 should have $(move_dims) rows and $(move_dims) columns")
    size(move.F)[1] == move_dims || println("DimensionMissmatch : Movement struct should have $(move_dims) rows")
    size(obsv.R)[1] == obsv_dims || println("DimensionMissmatch : Observation struct should have $(obsv_dims) rows")
    size(intr.K)[1] == move_dims || println("DimensionMissmatch : Interior.K should have $(move_dims) rows")
    size(obsv.H)[2] == move_dims || println("DimensionMissmatch : Observation.H should have $(move_dims) columns")
    length(intr.δ) == obsv_dims || println("DimensionMissmatch : Interior struct should have $(move_dims) rows and $(obsv_dims) columns")
end

"""
reset_kalman!(intr::Interior{T}, δ::Vector{T}, S::Matrix{T}, K::Matrix{T})
update the argument of Interior
"""
function reset_kalman! end

function reset_kalman!(intr::Interior{T}, δ::Vector{T}, S::Matrix{T}, K::Matrix{T}) where T <: Real
    intr.δ = δ
    intr.S = S
    intr.K = K 
end

"""
x_predict, P_predict = predict_kalman(x_update, P_update, move)
return the prediction of x and P when we know the movement infornmation.
"""

function predict_kalman end

function predict_kalman(x_update::Vector{T}, P_update::Matrix{T}, move::Movement{T}) where T <: Real
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
x_update, P_update = update_kalman!(x_predict, P_predict, record, obsv, Interior)
return the update for x and P, at the same time, update the arguments of Interior.
"""
function update_kalman! end

function update_kalman!(x_predict::Vector{T}, P_predict::Matrix{T}, record::Vector{T}, obsv::Observation{T}, intr::Interior{T}) where T <: Real
    δ = record - obsv.H * x_predict
    S = obsv.H * P_predict * transpose(obsv.H) + obsv.R
    K = P_predict * transpose(obsv.H) / S
    
    reset_kalman!(intr, δ, S, K)
    
    I = eye(length(x_predict), T)
    x_update = x_predict + K * δ
    P_update = (I - K * obsv.H) * P_predict
    
    return x_update, P_update
end

end