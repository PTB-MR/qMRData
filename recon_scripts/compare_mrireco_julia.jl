
using Pkg
using MRIReco, RegularizedLeastSquares, HDF5, NPZ

fname = "mri_reco_data.h5"
fid    = h5open(fname, "r")
kdat = read(fid["kdat"])
ktraj   = read(fid["ktraj"])
csm   = read(fid["csm"])
n_x0 = read(fid["n_x0"])
n_x1 = read(fid["n_x1"])
n_k0 = read(fid["n_k0"])
n_k1 = read(fid["n_k1"])
close(fid)


n_echoes = 1
n_slices = 1
n_repetitions = 1
n_coils = size(kdat, 2)

csm = reshape(ComplexF32.(csm), n_x0, n_x1, 1, n_coils)

tr = Trajectory(ktraj, n_k1, n_k0, circular=false)

kdata = Array{Matrix{ComplexF32}}(undef, n_echoes, n_slices, n_repetitions)
kdata[1, 1, 1] = ComplexF32.(kdat)


@show size(csm); flush(stdout)     
@show size(kdat); flush(stdout)          
@show size(tr.nodes); flush(stdout)      
@show n_coils; flush(stdout)
@show n_x0; flush(stdout)
@show n_x1; flush(stdout)
@show n_k0; flush(stdout)
@show n_k1; flush(stdout)

@show extrema(real(tr.nodes)); flush(stdout)  # should be in [-0.5, 0.5]
@show extrema(imag(tr.nodes)); flush(stdout)


acqData = AcquisitionData(
    Dict{Symbol,Any}(),                   
    [tr],                                 
    kdata,                                
    [collect(1:n_k0*n_k1)],                
    (n_x0, n_x1),           
    (1.0, 1.0, 1.0)                       
)

# Reconstruction
params = Dict{Symbol, Any}()
params[:reco] = "multiCoil"
params[:reconSize] = Tuple(acqData.encodingSize)
params[:iterations] = 1
params[:solver] = CGNR
params[:reg] = L2Regularization(0.0)
params[:senseMaps] = csm
img = reconstruction(acqData, params)

# Write reconstructed image to file
npzwrite("mri_reco_output.npz", Dict("img" => img))


