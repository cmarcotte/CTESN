using DifferentialEquations, ReservoirComputing, Surrogates, ModelingToolkit, Plots, MLJLinearModels, SparseArrays
#=

	To generate a CTESN for a stiff ODE system:
		- specify system model ODE
		- specify a non-special initial condition
		- specify upper and lower bounds for all parameters in model
		- select set of parameter vectors sampled from parameter distribution, p[i] ~ U(p_lower[i], p_upper[i])
		- generate initial solutions using stiff ODE model for each p[i] => u(t,p[i])
		- generate ESN ODE
		- train ESN ODE on each stiff ODE solution, p[i] => u(t,p[i]) => WOut(p[i])
		- interpolate set of WOut(p[i]) using Surrogates
		- test CTESN at new q ~ in {p[i]}_i

=#

# define original stiff ODE system
function robertson!(du, u, p, t)
	du[1] = -p[1] * u[1] + p[3] * u[2] * u[3]
	du[2] =  p[1] * u[1] - p[3] * u[2] * u[3] - p[2] * u[2]^2
	du[3] =                                     p[2] * u[2]^2
	return nothing
end

# define original parameters and initial condition
p0 = (0.04, 3e7, 1e4)
u0 = rand(Float64,3); u0 = u0./sum(u0) # to enforce sum(u) == 1.0
tspan = (0.0, 1e7)
tt = 10.0.^collect(range(-5.0, +5.0; length=33))
modelODESize = length(u0)

# define ODEProblem, optimize it, and solve for original ODE solution at (u0, p0)
prob = ODEProblem(robertson!, u0, tspan, p0)
#sol = solve(prob, Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=tt)
sys = modelingtoolkitize(prob)
f   = eval(ModelingToolkit.generate_function(sys)[2])
jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
prob = ODEProblem(ODEFunction(f, jac=jac), u0, tspan, p0)
sol = solve(prob, Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=tt)


reservoirSize = 3000
iterative = reservoirSize > 300

W   = sparse(ReservoirComputing.init_reservoir_givendeg(reservoirSize, 0.8, 6))
Win = ReservoirComputing.init_dense_input_layer(reservoirSize, modelODESize, 0.1)

function CTESN!(du, u, p, t)
	du .= tanh.(p[1]*u .+ p[2]*p[3](t))
	return nothing
end
esnprob = ODEProblem(CTESN!, randn(Float64, reservoirSize), tspan, (W, Win, sol))
r = solve(esnprob; saveat = sol.t)

@assert isapprox(sol.t, r.t) 

# compute first Wout
Wout = zeros(Float64, modelODESize, reservoirSize)
function fitData!(Wout, xData, rData; beta = 0.01, iterative = iterative)
	fitModel = RidgeRegression(beta) 
	#=
	Note:
		rData = hcat(r(sol.t)...)
		xData = hcat(sol(sol.t)...)
	But we avoid re-interpolating them every time by requiring the r solution use `saveat=sol.t`
	=#
	for n=1:modelODESize
		tmp = transpose(fit(fitModel, transpose(rData[:,:]), xData[n,:]; solver=Analytical(iterative=iterative)))
		Wout[n,:] .= tmp[1:reservoirSize]
	end
	return nothing
end
fitData!(Wout, sol[:,:], r[:,:]; beta=0.01)

# define lower and upper bounds for each parameter
p_lower = [0.036, 2.7e7, 0.9e4]
p_upper = [0.044, 3.3e7, 1.1e4]

# select set of Sobol-sampled parameter vectors
p	= sample(1000, p_lower, p_upper, SobolSample())
Wout 	= [zeros(Float64, modelODESize, reservoirSize) for _ in 1:length(p)]

# evaluate the ESN at each p, returning WOut(p)
function fitSamples!(Wout, p)
	
	function prob_func(prob, i, repeat)
		return remake(prob, p=p[i])
	end
	ensprob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensprob, Rosenbrock23(), EnsembleThreads(); trajectories=length(p), abstol=1e-6, reltol=1e-6, saveat=sol.t)
	for n in 1:length(p)
		print("Fitting p[$(n)]...   \t")
		fitData!(Wout[n], sim[n], r; beta = 1e-6)
		print("Done.\n")
	end
	return nothing
end
fitSamples!(Wout, p)
	
# form interpolant over sampled parameter vectors 
WoutInterpolant = RadialBasis(p, Wout, p_lower, p_upper)

# define unseen p̂ ensure it is not in p
p̂ = (0.0375,3.1e7, 1.01e4)
isinp = false;
for n in 1:length(p), m in 1:length(p̂)
	if isapprox(p̂[m], p[n][m])
		global isinp = true
		break
	end
end
@assert !isinp

# form Wout(p̂) and approximate x(t;p̂) from WoutInterpolant at unseen p̂
Ŵout = reshape(WoutInterpolant(p̂), size(Wout[1]))
x = Ŵout*r
#xInterpolant = RadialBasis(collect(transpose(sol.t)), x, [tspan[begin]], [tspan[end]])

sol_hat = solve(remake(prob, p=p̂), Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=sol.t)
p1 = plot(sol)
plot!(p1, sol_hat, linestyle=:dash, linewidth=2)
plot!(p1, sol.t, transpose(x), linestyle=:dot, linewidth=3)
plot!(p1, xscale=:log)
p2 = plot(sol.t, transpose(sum(sol,dims=1)).-1.0)
plot!(p2, sol_hat.t, transpose(sum(sol_hat,dims=1)).-1.0, linestyle=:dash, linewidth=2)
plot!(p2, sol.t, transpose(sum(x,dims=1).-1.0), linestyle=:dot, linewidth=3)
plot!(p2, xscale=:log)
plot(p1,p2,layout = (2, 1))
savefig("./ODE_newODE_newCTESN.svg")

