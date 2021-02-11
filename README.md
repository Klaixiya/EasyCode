# EasyCode

Something tiny and funny！

## Classic Kalman Filter

Now the KalmanFilter.jl just include the Classic Kalman Filter, so it is limited. 

All symbols are the same of   [wikipedia](https://en.wikipedia.org/wiki/Kalman_filter), so it is easy to contrast wiki when scanning the code.

let's see how it works.

### Initialize the model

```julia
using StatsBase
using Plots
using Distributions
include("KalmanFilter.jl")
const kf = KalmanFilter

x_0 = [0.0, 1.0]
P_0 = [0.6 0.0; 0.0 1.0]
F = [1.0 1.0; 0.0 1.0]
Q = 0.25 .* [0.0025 0.005; 0.005 0.01]
H = [1.0 0.0]
R = ones(1, 1) * 100.0

move = kf.Movement(F, Q)
obsv = kf.Observation(H, R)
interior = kf.Interior(2, 1)

```

### generating data

```julia
xprocess = x_0
xnew = x_0
for i in 1:200
    xnew = F*xnew + [0.5, 1.0]*rand(Normal(0.0, 0.05))
    xprocess = hcat(xprocess, xnew)
end
processlist = xprocess[1, 2:end]
randomnoise = rand(Normal(0, 10), 200)
recordlist = processlist + randomnoise
recordlist = reshape(recordlist, 1, length(recordlist))
```

### check dimension

```julia
kf.check_kalman(x, P, recordlist[:, 1], interior, move, obsv)
```

### trace the observing data

```julia
x_update, P_update = x_0, P_0
xlist = x
Klist = [0.0; 0.0]
for i in 1:size(recordlist)[2]
    x_predict, P_predict = kf.predict_kalman(x_update, P_update, move)
    x_update, P_update = kf.update_kalman!(x_predict, P_predict, recordlist[:, i], obsv, interior)
    xlist = hcat(xlist, x_update)
    Klist = hcat(Klist, interior.K)
end
```

### data visualization

```julia
p2 = plot(processlist, label = "real trace")
plot!(p2, xlist[1, 2:end], label = "km trace")
plot!(p2, recordlist[1, :], label = "obseved trace", legend = :topleft, dpi = 150)

p3 = plot(xlist[2, 2:end], label = "km velocity")
plot!(p3, xprocess[2, 2:end], label = "real velocity", dpi = 150)

p1 = plot(processlist .- xlist[1, 2:end])
plot!(p1, processlist .- recordlist[1, :], dpi = 150)

plot(p2, p3, p1, layout = (3, 1), dpi = 150, legend = false)
```

![simulation](images/simuliation.png)



## Unscented Kalman Filter

### Initialize the model

```julia
using Plots
using Statistics
using Distributions
using LinearAlgebra

x0 = [10., 0., 0., 1.]
δt = 0.5
σq = 0.01
ω = 0.1
F = [1.     δt      0.      0.  
     -ω^2   1.      0.      0.
     0.     0.      1.     δt
     0.     0.      -ω^2   1.]

Q = [0.25*δt^4  0.5*δt^3      0.      0.
     0.5*δt^3     δt^2        0.      0.
     0.           0.    0.25*δt^4  0.5*δt^3
     0.           0.    0.5*δt^3     δt^2] * σq^2
Ftransform(x::Vector{Float64}) = F*x
Htransform(x::Vector{Float64}) = [√(x[1]^2 + x[3]^2), atan(x[3], x[1])]
H_inverse(x::Vector{Float64}) = [x[1]*cos(x[2]), x[1]*sin(x[2])]
σl = 0.5
σθ = 0.05
R = [σl^2  0.
     0.   σθ^2]
```



### generate true trace and observed trace

```julia
Steps = 200
pos = x0
true_state = x0
for i in 1:Steps
    pos = F*pos + [0.5*δt^2, δt, 0., 0.] * rand(Normal(0., σq)) + [0., 0., 0.5*δt^2, δt] * rand(Normal(0., σq))
    true_state = hcat(true_state, pos)
end
true_trace = true_state[[1, 3], :]

observe_state = mapslices(Htransform, true_state, dims = 1)
observe_state = observe_state + rand(MvNormal(R), Steps + 1)
observe_trace = mapslices(H_inverse, observe_state, dims = 1)

p1 = plot(true_trace[1, :], true_trace[2, :], label = "true trace")
plot!(p1, observe_trace[1, :],observe_trace[2, :], label = "observe trace", dpi = 150, images
```

![true trace and observed trace](images/unscented kalman filter img0.png)



### unscented kalman filter

```julia
include("KalmanFilter.jl")
const kf = KalmanFilter

sigma = kf.Sigma(1., 2., 2.)
noise = kf.Noise(Q, R)

x_update = [11.2, 0., 0., 1.2]
P_update = [1.2  0.   0.   0.
            0.  0.2   0.   0.
            0.  0.    1.2  0.
            0.  0.    0.   0.2]
xlist = x0
for i in 2:Steps+1
    x_update, P_update, K = noise(x_update, P_update, observe_state[:, i], sigma, Ftransform, Htransform)
    xlist = hcat(xlist, x_update)
end

plot!(p1, xlist[1, :], xlist[3, :], label = "filter trace", dpi = 150, legend = :topleft)
```

![compare filter trace](images/unscented kalman filter img1.png)



*Let's see the errors of filter trace and observed trace*

![error](images/unscented kalman filter img2.png)



