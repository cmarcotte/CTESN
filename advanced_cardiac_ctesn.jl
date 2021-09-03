using DifferentialEquations, ReservoirComputing, Surrogates, ModelingToolkit, Plots, MLJLinearModels, SparseArrays, ProgressBars, CellMLToolkit, Sundials

model_root = joinpath(splitdir(pathof(CellMLToolkit))[1], "..", "models")
ml = CellModel(joinpath(model_root, "ohara_rudy_cipa_v1_2017.cellml.xml"))

tspan = (0, 2000.0)
prob = ODEProblem(ml, tspan);
sol = solve(prob, Rosenbrock23(autodiff=false), dtmax=0.5, save_idxs=49)
plot(sol, vars=1)

#p = list_params(ml)
#update_list!(p, :membrane₊i_Stim_Period, 160.0)
#prob = ODEProblem(ml, tspan; p=p)
p = prob.p; p[201] = 300.0; prob = remake(prob, p=p)
tt = collect(tspan[1]:1.0:tspan[2])
sol = solve(prob, Rosenbrock23(autodiff=false), dtmax=0.5, save_idxs=49:49, saveat=tt)
plot!(sol, vars=1)

modelODESize = 1 #length(prob.u0) # 1 => just fit one component; length(prob.u0) => fit em all

reservoirSize = 300
iterative = reservoirSize > 300 # this is kind of machine-specific?

W   = ReservoirComputing.init_reservoir_givendeg(reservoirSize, 0.8, 6)
Win = ReservoirComputing.init_dense_input_layer(reservoirSize, modelODESize, 0.1)
if iterative
	W = sparse(W)
end

function CTESN!(du, u, p, t)
	du .= p[1].*tanh.(p[2]*u .+ p[3]*p[4](t; idxs=1:1)) .- (1.0 - p[1]).*u # p[3](t; idxs=49:49)
	return nothing
end
esnprob = ODEProblem(CTESN!, randn(Float64, reservoirSize), tspan, (0.8, W, Win, sol))
r = solve(esnprob; saveat = sol.t)

@assert isapprox(sol.t, r.t) 

# compute first Wout
Wout = zeros(Float64, modelODESize, reservoirSize)
function fitData!(Wout, xData, rData; beta = 1e-3, iterative = iterative)
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
fitData!(Wout, sol[1:1,:], r[:,:]; beta=1e-3)

# define lower and upper bounds for each variable
#u_lower = minimum(sol, dims=2)[49:49]
#u_upper = maximum(sol, dims=2)[49:49]
p_lower = (40.0)
p_upper = (1000.0)

# select set of Sobol-sampled state vectors
P	= sample(200, p_lower, p_upper, SobolSample()) # this generates tuples; convert to arrays in prob_func
Wout 	= [zeros(Float64, modelODESize, reservoirSize) for _ in 1:length(P)]

# evaluate the ESN at each p, returning WOut(p)
function fitSamples!(Wout, P; p0=p, beta=1e-3)
	
	function prob_func(prob, i, repeat)
		p0[201] = P[i][1]
		println(i)
		return remake(prob, p=p0)
	end
	ensprob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensprob, Rosenbrock23(autodiff=false), EnsembleThreads(); trajectories=length(P), dtmax=0.5, save_idxs=49:49, saveat=tt)
	for n in ProgressBar(1:length(P))
		#print("Fitting u[$(n)]...   \t")
		fitData!(Wout[n], sim[n][1:1,:], r; beta = beta)
		#print("Done.\n")
	end
	return nothing
end
fitSamples!(Wout, P)

# form interpolant over sampled parameter vectors 
WoutInterpolant = RadialBasis(P, Wout, p_lower, p_upper)

# define unseen p̂ ensure it is not in p
p̂ = (125.0)
# form Wout(p̂) and approximate x(t;p̂) from WoutInterpolant at unseen p̂
Ŵout = reshape(WoutInterpolant(p̂), size(Wout[1]))
x = Ŵout*r
#xInterpolant = RadialBasis(collect(transpose(sol.t)), x, [tspan[begin]], [tspan[end]])
p0 = p
p0[201] = p̂[1]
sol_hat = solve(remake(prob, p=p0), Rosenbrock23(autodiff=false); dtmax=0.5, save_idxs=49:49, saveat=sol.t)

plt = plot(sol, vars=(0,1))
plot!(plt, sol_hat, vars=(0,1), linestyle=:dash, linewidth=2)
plot!(plt, sol.t, x[1,:], linestyle=:dot, linewidth=3)
plot!(legend=false)
#if n==2; plot!(plt, yscale=:log); end
plot(plt)
savefig("./ohara_rudy_CTESN.svg")

