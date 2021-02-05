# This module export kalman filter structures and function.

# just for simple Linear system.

#=
The normal workflw is like the:

```julia

# initialize all arguments
x0, P0 = x0, P0 
record_list = [...]
interior = Interior{T}(row, collumn)
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
 
export Kalmanfilter, Interior1d, Interior, Movement, Observation1d, Observation, predict_kalman, update_kalman!

"""
abstract struct Kalmanfilter include 3 structures.

    Interior : storage the filter Interior arguments
    Movement : storage the Linear system movement arguments
    Observation: storage the transform from movement system to Observation and noise of Observation.
"""
abstract type Kalmanfilter end

"""
Interior1d is simple version for Interior when the Observation is a value nor a vector.
"""
mutable struct Interior1d{T <: Real} <: Kalmanfilter
    δ::T
    S::T
    K::Vector{T}

end
Interior1d(row::Int, T::DataType = Float64) = Interior1d{T}(0.0, 0.0, zeros(T, row))

"""
Interior explain how the estimation is done:
    δ : Observation - predict_of_movement
    S : pre-fit residual
    K : Optimal Kalman gain
"""
mutable struct Interior{T <: Real} <: Kalmanfilter
    δ::Vector{T}
    S::Matrix{T}
    K::Matrix{T}
end
Interior(row::Int, column::Int, T::DataType = Float64) = Interior{T}(zeros(T, row), zeros(T, row, column), zeros(T, row, column))

"""
Movement show how the system moves according to matrix (Linear System)
    F : x_k = F * x_k-1 + B * u_k + w_k
    Q : covariance of w_k (noise strength and interact)
    B : count the influence of Input u_k to the movement
    u : Input, maybe a control variable
"""
mutable struct Movement{T <: Real} <: Kalmanfilter
    F::Matrix{T}
    Q::Matrix{T}
    B::Matrix{T}
    u::Vector{T}
    Movement(F, Q, B, u) = (size(F) == size(Q) && size(B) == (size(F)[1], length(u))) ? new{eltype(u)}(F, Q, B, u) : error("Dimension does't matc")
end
Movement(F::Matrix{T}, Q::Matrix{T}) where T <: Real = Movement(F, Q, zeros(T, size(Q)...), zeros(T, size(Q)[1]))

"""
Observation1d is simple version for Observation when the Observation is a value nor a vector.
"""
mutable struct Observation1d{T <: Real} <: Kalmanfilter
    H::Matrix{T}
    R::T
end
Observation1d(H::Matrix{T}, R::T) where T <: Real = Observation1d{T}(H, R)

"""
Observation tells how the state of movement tranfer to Observation.
    H : Observation = H * x
    R : covariance of observing noise.
"""
mutable struct Observation{T <: Real} <: Kalmanfilter
    H::Matrix{T}
    R::Matrix{T}
    Observation(H, R) = (size(H)[2] == size(R)[1]) ? new{eltype(H)}(H, R) : error("Dimension does't match!")
end
Observation(H::Matrix{T}, R::Matrix{T}) where T <: Real = Observation(H, R)


"""
x_predict, P_predict = predict_kalman(x_update, P_update, move)
return the prediction of x and P when we know the movement infornmation.
"""
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

    return the update for x and P, at the same time, update the arguments of interior.
    the Observation has just a value (record).
"""
function update_kalman!(x_predict::Vector{T}, P_predict::Matrix{T}, record::T, obsv::Observation1d{T}, interior::Interior1d{T}) where T <: Real
    δ = record - (obsv.H * x_predict)[1]
    S = (obsv.H * P_predict * transpose(obsv.H))[1] + obsv.R
    K = (P_predict * transpose(obsv.H))[:, 1] / S
    
    interior.δ = δ
    interior.S = S
    interior.K = K
    
    I = eye(length(x_predict), T)
    x_update = x_predict + K * δ
    P_update = (I - K * obsv.H) * P_predict
    
    return x_update, P_update
end

"""
x_update, P_update = update_kalman!(x_predict, P_predict, record, obsv, Interior)
return the update for x and P, at the same time, update the arguments of interior.
"""
function update_kalman!(x_predict::Vector{T}, P_predict::Matrix{T}, record::Vector{T}, obsv::Observation{T}, Interior::Interior{T}) where T <: Real
    δ = record - obsv.H * x_predict
    S = obsv.H * P_predict * transpose(obsv.H) + obsv.R
    K = P_predict * transpose(obsv.H) / S
    
    Interior.δ = δ
    Interior.S = S
    Interior.K = K
    
    I = eye(Tlength(x_predict), T)
    x_update = x_predict + K * δ
    P_update = (I - K * H) * P_predict
    
    return x_update, P_update
end

end