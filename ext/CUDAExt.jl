module CUDAExt

export CuArrayDeviceProc

import Dagger, DaggerGPU, MemPool
import Dagger: CPURAMMemorySpace, Chunk
import MemPool: DRef, poolget
import Distributed: myid, remotecall_fetch
import LinearAlgebra
using KernelAbstractions, Adapt

const CPUProc = Union{Dagger.OSProc,Dagger.ThreadProc}

if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end
import CUDA: CuDevice, CuContext, CuArray, CUDABackend, devices, attribute, context, context!
import CUDA: CUBLAS, CUSOLVER

using UUIDs

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    device::Int
    device_uuid::UUID
end
Dagger.get_parent(proc::CuArrayDeviceProc) = Dagger.OSProc(proc.owner)
Dagger.root_worker_id(proc::CuArrayDeviceProc) = proc.owner
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc(worker $(proc.owner), device $(proc.device), uuid $(proc.device_uuid))")
Dagger.short_name(proc::CuArrayDeviceProc) = "W: $(proc.owner), CUDA: $(proc.device)"
DaggerGPU.@gpuproc(CuArrayDeviceProc, CuArray)

"Represents the memory space of a single CUDA GPU's VRAM."
struct CUDAVRAMMemorySpace <: Dagger.MemorySpace
    owner::Int
    device::Int
    device_uuid::UUID
end
Dagger.root_worker_id(space::CUDAVRAMMemorySpace) = space.owner
function Dagger.memory_space(x::CuArray)
    dev = CUDA.device(x)
    device_id = dev.handle
    device_uuid = CUDA.uuid(dev)
    return CUDAVRAMMemorySpace(myid(), device_id, device_uuid)
end

Dagger.memory_spaces(proc::CuArrayDeviceProc) = Set([CUDAVRAMMemorySpace(proc.owner, proc.device, proc.device_uuid)])
Dagger.processors(space::CUDAVRAMMemorySpace) = Set([CuArrayDeviceProc(space.owner, space.device, space.device_uuid)])

unwrap(x::Chunk) = MemPool.poolget(x.handle)
# TODO: No extra allocations here
to_device(proc::CuArrayDeviceProc) = collect(CUDA.devices())[proc.device+1]
with_context!(ctx::CuContext) = context!(ctx)
with_context!(proc::CuArrayDeviceProc) = with_context!(context(to_device(proc)))
with_context!(x::CuArray) = with_context!(context(x))
function with_context(f, x)
    old_ctx = context()
    with_context!(x)
    try
        f()
    finally
        context!(old_ctx)
    end
end

# In-place
function Dagger.move!(to_space::Dagger.CPURAMMemorySpace, from_space::CUDAVRAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    if from isa CuArray
        with_context!(from)
        CUDA.synchronize()
    end
    copyto!(to, from)
    if from isa CuArray
        CUDA.synchronize()
    end
    return
end
function Dagger.move!(to_space::CUDAVRAMMemorySpace, from_space::Dagger.CPURAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    if to isa CuArray
        with_context!(to)
    end
    copyto!(to, from)
    if to isa CuArray
        CUDA.synchronize()
    end
    return
end
function Dagger.move!(to_space::CUDAVRAMMemorySpace, from_space::CUDAVRAMMemorySpace, to::AbstractArray{T,N}, from::AbstractArray{T,N}) where {T,N}
    with_context(CUDA.synchronize, from)
    with_context(to) do
        copyto!(to, from)
        CUDA.synchronize()
    end
    return
end

# function can_access(this, peer)
#     status = Ref{Cint}()
#     CUDA.cuDeviceCanAccessPeer(status, this, peer)
#     return status[] == 1
# end

# Out-of-place HtoD
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x)
    with_context(to_proc) do
        arr = adapt(CuArray, x)
        CUDA.synchronize()
        return arr
    end
end
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Chunk)
    from_w = Dagger.root_worker_id(from_proc)
    to_w = Dagger.root_worker_id(to_proc)
    @assert myid() == to_w
    cpu_data = remotecall_fetch(unwrap, from_w, x)
    with_context(to_proc) do
        arr = adapt(CuArray, cpu_data)
        CUDA.synchronize()
        return arr
    end
end
function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::CuArray)
    if device(x) == to_device(to_proc)
        return x
    end
    with_context(to_proc) do
        _x = similar(x)
        copyto!(_x, x)
        CUDA.synchronize()
        return _x
    end
end

# Out-of-place DtoH
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x)
    with_context(from_proc) do
        CUDA.synchronize()
        return adapt(Array, x)
    end
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x::Chunk)
    from_w = Dagger.root_worker_id(from_proc)
    to_w = Dagger.root_worker_id(to_proc)
    @assert myid() == to_w
    remotecall_fetch(from_w, x) do x
        arr = unwrap(x)
        return Dagger.move(from_proc, to_proc, arr)
    end
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x::CuArray{T,N}) where {T,N}
    with_context(x) do
        CUDA.synchronize()
        _x = Array{T,N}(undef, size(x))
        copyto!(_x, x)
        return _x
    end
end

# Out-of-place DtoD
function Dagger.move(from::CuArrayDeviceProc, to::CuArrayDeviceProc, x::Dagger.Chunk{T}) where T<:CuArray
    if from == to
        # Same process and GPU, no change
        arr = unwrap(x)
        with_context(CUDA.synchronize, context(arr))
        return arr
    elseif from.owner == to.owner
        # Same process but different GPUs, use DtoD copy
        from_arr = unwrap(x)
        with_context(CUDA.synchronize, context(from_arr))
        return with_context(to) do
            to_arr = similar(from_arr)
            copyto!(to_arr, from_arr)
            CUDA.synchronize()
            to_arr
        end
    elseif Dagger.system_uuid(from.owner) == Dagger.system_uuid(to.owner)
        error("TODO: Remove me")
        # Same node, we can use IPC
        ipc_handle, eT, shape = remotecall_fetch(from.owner, x) do x
            arr = unwrap(x)
            ipc_handle_ref = Ref{CUDA.CUipcMemHandle}()
            GC.@preserve arr begin
                CUDA.cuIpcGetMemHandle(ipc_handle_ref, pointer(arr))
            end
            (ipc_handle_ref[], eltype(arr), size(arr))
        end
        r_ptr = Ref{CUDA.CUdeviceptr}()
        CUDA.device!(from.device) do # FIXME: Assumes that device IDs are identical across processes
            CUDA.cuIpcOpenMemHandle(r_ptr, ipc_handle, CUDA.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        end
        ptr = Base.unsafe_convert(CUDA.CuPtr{eT}, r_ptr[])
        arr = unsafe_wrap(CuArray, ptr, shape; own=false)
        finalizer(arr) do arr
            CUDA.cuIpcCloseMemHandle(pointer(arr))
        end
        if from.device_uuid != to.device_uuid
            return CUDA.device!(to.device) do
                to_arr = similar(arr)
                copyto!(to_arr, arr)
                to_arr
            end
        else
            return arr
        end
    else
        error("TODO: Remove me")
        # Different node, use DtoH, serialization, HtoD
        # TODO UCX
        return CuArray(remotecall_fetch(from.owner, x.handle) do h
            Array(MemPool.poolget(h))
        end)
    end
end

# Adapt generic functions
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::Chunk{T}) where {T<:Function} =
    Dagger.move(from_proc, to_proc, fetch(x))

# Adapt BLAS/LAPACK functions
# TODO: Create these automatically
import LinearAlgebra: BLAS, LAPACK
import .LAPACK: potrf!
for fn in [potrf!]
    Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, ::typeof(fn)) = getproperty(CUSOLVER, nameof(fn))
end
function potrf_checked!(uplo, A, info_arr)
    @assert context() === context(A)
    was_posdef = LinearAlgebra.isposdef(A)
    if !was_posdef
        sleep(1)
    end
    @show (was_posdef, LinearAlgebra.isposdef(A))
    _A, info = CUSOLVER.potrf!(uplo, A)
    @assert context() === context(_A)
    @show info
    if info > 0
        fill!(info_arr, info)
        throw(LinearAlgebra.PosDefException(info))
    end
    return _A, info
end
Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, ::typeof(Dagger.potrf_checked!)) = potrf_checked!
import .BLAS: trsm!, syrk!, gemm!
for fn in [trsm!, syrk!, gemm!]
    Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, ::typeof(fn)) = getproperty(CUBLAS, nameof(fn))
end

# Task execution
function Dagger.execute!(proc::CuArrayDeviceProc, f, args...; kwargs...)
    @nospecialize f args kwargs
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        dev = to_device(proc)
        ctx = context(dev)
        # FIXME: Remove me
        for arg in args
            if arg isa CuArray
                if f != Dagger.move!
                    if ctx != context(arg)
                        println("WARN: mismatched context ($ctx vs arg $(context(arg))), ($dev vs arg $(CUDA.device(arg))), type $(typeof(arg))")
                    end
                    @assert ctx == context(arg)
                end
                with_context(CUDA.synchronize, arg)
            end
        end
        with_context!(ctx)
        result = Base.@invokelatest f(args...; kwargs...)
        CUDA.synchronize()
        return result
    end

    try
        fetch(task)
    catch err
        stk = current_exceptions(task)
        err, frames = stk[1]
        rethrow(CapturedException(err, frames))
    end
end

DaggerGPU.processor(::Val{:CUDA}) = CuArrayDeviceProc
DaggerGPU.cancompute(::Val{:CUDA}) = CUDA.has_cuda()
DaggerGPU.kernel_backend(::CuArrayDeviceProc) = CUDABackend()
DaggerGPU.with_device(f, proc::CuArrayDeviceProc) =
    CUDA.device!(f, proc.device)

Dagger.to_scope(::Val{:cuda_gpu}, sc::NamedTuple) =
    Dagger.to_scope(Val{:cuda_gpus}, merge(sc, (;cuda_gpus=[sc.cuda_gpu])))
Dagger.scope_key_precedence(::Val{:cuda_gpu}) = 1
function Dagger.to_scope(::Val{:cuda_gpus}, sc::NamedTuple)
    if haskey(sc, :worker)
        workers = Int[sc.worker]
    elseif haskey(sc, :workers) && sc.workers != Colon()
        workers = sc.workers
    else
        # FIXME: Check context
        workers = map(gproc->gproc.pid, Dagger.procs(Dagger.Sch.eager_context()))
    end
    scopes = Dagger.ExactScope[]
    dev_ids = sc.cuda_gpus
    for worker in workers
        procs = Dagger.get_processors(Dagger.OSProc(worker))
        for proc in procs
            proc isa CuArrayDeviceProc || continue
            if dev_ids == Colon() || proc.device+1 in dev_ids
                scope = Dagger.ExactScope(proc)
                push!(scopes, scope)
            end
        end
    end
    return Dagger.UnionScope(scopes)
end
Dagger.scope_key_precedence(::Val{:cuda_gpus}) = 1

function __init__()
    if CUDA.has_cuda()
        for dev in CUDA.devices()
            @debug "Registering CUDA GPU processor with Dagger: $dev"
            Dagger.add_processor_callback!("cuarray_device_$(dev.handle)") do
                CuArrayDeviceProc(myid(), dev.handle, CUDA.uuid(dev))
            end
        end
    end
end

end # module CUDAExt
