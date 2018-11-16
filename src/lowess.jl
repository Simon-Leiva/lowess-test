
using DataFrames
using LinearAlgebra

    # We define a lowess model object
mutable struct lowessModel{T<:Vector{Float64}, K<:Float64}
    x::T # array with predictors
    y::T # array with responses
    wSize::K # Size if the window

#    function lowessModel(x::T, y::T, wSize::K) where {T, K}
#        if length(y) != length(x)
#            error("Length of predictors are not the same as the responses")
#        end
#        if wSize <= 0
#            error("Window size must be greater than zero")
#        end
#    end
end

# This function is used to create the model matrix
function RegMatrix(obj::lowessModel)
    n = length(obj.x)
    hcat(ones(n), obj.x)
end
# Epanechnikov Kernel
function EpaKernel(t)
    (abs(t) <= 1) * (1 - abs(t) ^ 3) ^ 3
end

function Tricube(x0, x, l)
    EpaKernel(sqrt((x .- x0) .^ 2) ./ l)
end

# used to create the weight matrix
function WeightMatrix(obj::lowessModel, x0)
    diagm(0 => Tricube.(x0, obj.x, obj.wSize))
end

# Fit lowess for one observation
function fitLowess(obj::lowessModel, i)
    x0 = obj.x[i]
    W = WeightMatrix(obj, x0)
    X = RegMatrix(obj)
    y = obj.y
    X[i, :]' * inv(X' * W * X) * X' * W * y
end

# Mapper to all observations
function Lowess(obj::lowessModel)
    n = size(obj.y)[1]
    map(i -> fitLowess(obj, i), collect(1:1:(n)))
end

# Some simple tests

using Distributions, Plots

x = collect(1:1:1000)
y = sin.(x/50) * 1000 .+ rand(Normal(0, 300), 1000)
scatter(x, y)

myModel = lowessModel(x * 1.0, y, 50.0)
preds = Lowess(myModel)

scatter(x, y)
plot!(x, preds, color = "red", linewidth = 4)
