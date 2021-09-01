using DifferentialEquations, ReservoirComputing, Surrogates, ModelingToolkit, Plots, MLJLinearModels, SparseArrays, ProgressBars

function ab(C::Array{Float64,1},V)
	# eq (13) from original paper
	return (C[1]*exp(C[2]*(V+C[3]))+C[4]*(V+C[5]))/(exp(C[6]*(V+C[3]))+C[7])
end

function Casmoother(Ca; ep=1.5e-10)
	return ep*0.5*(1.0 + tanh(1.0-(Ca/ep))) + Ca*0.5*(1.0 + tanh((Ca/ep)-1.0))
end

function BR!(dx,x,p,t)
	
	# spatially local currents
	IK = (exp(0.08*(x[1]+53.0)) + exp(0.04*(x[1]+53.0)))
	IK = 4.0*(exp(0.04*(x[1]+85.0)) - 1.0)/IK
	IK = IK+0.2*(x[1]+23.0)/(1.0-exp(-0.04*(x[1]+23.0)))
	IK = 0.35*IK
	Ix = x[3]*0.8*(exp(0.04*(x[1]+77.0))-1.0)/exp(0.04*(x[1]+35.0))
	INa= (4.0*x[4]*x[4]*x[4]*x[5]*x[6] + 0.003)*(x[1]-50.0)
	Is = 0.09*x[7]*x[8]*(x[1]+82.3+13.0287*log(x[2]))#Casmoother(x[2])))

	# these from Beeler & Reuter table:
	ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],x[1])
	bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],x[1])
	am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],x[1])
	bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],x[1])
	ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],x[1])
	bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],x[1])
	aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],x[1])
	bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],x[1])
	ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],x[1])
	bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],x[1])
	af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],x[1])
	bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],x[1])

	# BR dynamics
	dx[1] = -(IK + Ix + INa + Is)/p[1]
	dx[2] = -10^-7 * Is + 0.07*(10^-7 - x[2])
	dx[3] = ax*(1.0 - x[3]) - bx*x[3]
	dx[4] = am*(1.0 - x[4]) - bm*x[4]
	dx[5] = ah*(1.0 - x[5]) - bh*x[5]
	dx[6] = aj*(1.0 - x[6]) - bj*x[6]
	dx[7] = ad*(1.0 - x[7]) - bd*x[7]
	dx[8] = af*(1.0 - x[8]) - bf*x[8]
	
	return nothing

end

# define original parameters and initial condition
# initial condition
u0 = [ -60.0,10^-7,0.01,0.01,0.99,0.99,0.01,0.99]
# parameters 
p0 = (1.0)
tspan = (0.0, 1e3)
tt = 10.0.^collect(range(-3.0, +3.0; length=129))
modelODESize = length(u0)

# define ODEProblem, optimize it, and solve for original ODE solution at (u0, p0)
prob = ODEProblem(BR!, u0, tspan, p0)
#sol = solve(prob, Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=tt)
sys = modelingtoolkitize(prob)
f   = eval(ModelingToolkit.generate_function(sys)[2])
jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
prob = ODEProblem(ODEFunction(f, jac=jac), u0, tspan, p0)
sol = solve(prob, Rosenbrock23(); abstol=1e-10, reltol=1e-8, saveat=tt)


reservoirSize = 300
iterative = reservoirSize > 300 # this is kind of machine-specific?

W   = ReservoirComputing.init_reservoir_givendeg(reservoirSize, 0.8, 6)
Win = ReservoirComputing.init_dense_input_layer(reservoirSize, modelODESize, 0.1)
if iterative
	W = sparse(W)
end

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

# define lower and upper bounds for each variable
u_lower = minimum(sol, dims=2)
u_upper = maximum(sol, dims=2)
#u_lower = u0 .+ 1e-1.*abs.(u0)
#u_upper = u0 .- 1e-1.*abs.(u0)

# select set of Sobol-sampled state vectors
u	= sample(1000, u_lower, u_upper, SobolSample()) # this generates tuples; convert to arrays in prob_func
Wout 	= [zeros(Float64, modelODESize, reservoirSize) for _ in 1:length(u)]

# evaluate the ESN at each p, returning WOut(u)
function fitSamples!(Wout, u)
	
	function prob_func(prob, i, repeat)
		return remake(prob, u0=[u[i][j] for j=1:8])
	end
	ensprob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensprob, Rosenbrock23(), EnsembleThreads(); trajectories=length(u), abstol=1e-10, reltol=1e-8, saveat=sol.t)
	for n in ProgressBar(1:length(u))
		#print("Fitting u[$(n)]...   \t")
		fitData!(Wout[n], sim[n], r; beta = 1e-6)
		#print("Done.\n")
	end
	return nothing
end
fitSamples!(Wout, u)
	
# form interpolant over sampled parameter vectors 
WoutInterpolant = RadialBasis(u, Wout, u_lower, u_upper)

# define unseen p̂ ensure it is not in p
û= u0
issampled = false;
for n in 1:length(u), m in 1:length(û)
	if isapprox(û[m], u[n][m])
		global issampled = true
		break
	end
end
Ŵout = reshape(WoutInterpolant(û), size(Wout[1]))
x = Ŵout*r
if issampled
	û= u0 .+ 1e-3 .* randn(size(u0)) .* u0
	# form Wout(p̂) and approximate x(t;p̂) from WoutInterpolant at unseen p̂
	Ŵout = reshape(WoutInterpolant(û), size(Wout[1]))
	x = Ŵout*r
	#xInterpolant = RadialBasis(collect(transpose(sol.t)), x, [tspan[begin]], [tspan[end]])

	sol_hat = solve(remake(prob, u0=û), Rosenbrock23(); abstol=1e-6, reltol=1e-6, saveat=sol.t)
	
end

pp = []
for n=1:length(u0)
	plt = plot(sol, vars=(0,n))
	if issampled; plot!(plt, sol_hat, vars=(0,n), linestyle=:dash, linewidth=2); end
	plot!(plt, sol.t, x[n,:], linestyle=:dot, linewidth=3)
	plot!(plt, xscale=:log)
	plot!(legend=false)
	#if n==2; plot!(plt, yscale=:log); end
	push!(pp, plt)
end
plot(pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], size=(1000,500), layout=(2,4))
savefig("./BR_CTESN.svg")

