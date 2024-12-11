use std::{ffi::{c_void, CString}, ptr};

use cuda_sys::cuda::*;

pub fn cuda_init() -> Result<CUcontext, CUresult> {
    // Initialize the CUDA driver
    let init_result = unsafe { cuInit(0) };
    if init_result != cudaError_t::CUDA_SUCCESS {
        return Err(init_result);

    }

    // Get the first available device
    let mut device: CUdevice = 0;
    let device_result = unsafe { cuDeviceGet(&mut device, 0) };
    if device_result != cudaError_t::CUDA_SUCCESS {
        return Err(device_result);

    }

    // Create a context for the device
    let mut context: CUcontext = ptr::null_mut();
    let context_result = unsafe { cuCtxCreate_v2(&mut context, 0, device) };
    if context_result == cudaError_t::CUDA_SUCCESS {
        Ok(context)  // Return the context if successful

    } else {
        Err(context_result)  // Return the error code on failure

    }

}

pub fn device_mem_alloc<T>(size: usize) -> Result<CUdeviceptr, CUresult> {
    let mut device_ptr: CUdeviceptr = CUdeviceptr::default();

    let result: cudaError_t = unsafe {
        cuMemAlloc_v2(
            &mut device_ptr,
            size * std::mem::size_of::<T>(),

        )

    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(device_ptr)

    } else {
        Err(result)

    }

}

pub fn device_mem_free(device_ptr: CUdeviceptr) -> Result<(), CUresult> {
    let result: cudaError_t = unsafe { cuMemFree_v2(device_ptr as CUdeviceptr) };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn allocate_zero_copy_memory<T>(size: usize) -> Result<(*mut T, CUdeviceptr), CUresult> {
    let mut host_ptr: *mut c_void = std::ptr::null_mut();
    let bytes = size * std::mem::size_of::<T>();

    // Allocate pinned host memory
    let result: cudaError_t = unsafe {
        cuMemHostAlloc(
            &mut host_ptr as *mut *mut c_void,
            bytes,
            CU_MEMHOSTALLOC_DEVICEMAP
        )
    };
    if result != cudaError_t::CUDA_SUCCESS {
        return Err(result);

    }

    // Retrieve the device pointer for the mapped memory
    let mut device_ptr: CUdeviceptr = 0;
    unsafe { cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0) };

    Ok((host_ptr as *mut T, device_ptr))

}

pub fn mem_copy_to_device<T>(device_ptr: CUdeviceptr, host_data: &[T]) -> Result<(), CUresult> {
    let size_in_bytes = host_data.len() * std::mem::size_of::<T>();

    let result: cudaError_t = unsafe {
        cuMemcpyHtoD_v2(
            device_ptr,
            host_data.as_ptr() as *const std::ffi::c_void,
            size_in_bytes

        )

    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn mem_copy_to_host<T>(device_ptr: CUdeviceptr, host_data: *mut T, size: usize) -> Result<(), CUresult> {
    let size_in_bytes = size * std::mem::size_of::<T>();

    let result: cudaError_t = unsafe {
        cuMemcpyDtoH_v2(host_data as *mut c_void, device_ptr, size_in_bytes)
    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(())

    } else {
        Err(result)

    }

}

pub fn load_module_from_ptx(ptx: &str) -> Result<CUmodule, cudaError_t> {
    let c_file_path = CString::new(ptx).expect("Failed to convert PTX to CString");

    let mut module: CUmodule = ptr::null_mut();

    let result: cudaError_t = unsafe {
        cuModuleLoad(&mut module, c_file_path.as_ptr())
    };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(module)

    } else {
        Err(result)

    }

}

pub fn get_kernel_function(module: CUmodule, kernel_name: &str) -> Result<CUfunction, CUresult> {
    let c_kernel_name = CString::new(kernel_name).expect("Failed to create CString for kernel name");

    let mut function: CUfunction = ptr::null_mut();

    let result: cudaError_t = unsafe { cuModuleGetFunction(&mut function, module, c_kernel_name.as_ptr()) };

    if result == cudaError_t::CUDA_SUCCESS {
        Ok(function)

    } else {
        Err(result)

    }

}

pub unsafe fn set_symbol_device_pointer(module: CUmodule, symbol_name: &str, device_ptr: CUdeviceptr) -> Result<(), CUresult> {
    let c_symbol_name = CString::new(symbol_name).expect("Symbol name conversion failed");

    let mut symbol_address: CUdeviceptr = 0;
    let mut symbol_size: usize = 0;

    let result = cuModuleGetGlobal_v2(
        &mut symbol_address as *mut CUdeviceptr,
        &mut symbol_size as *mut usize,
        module,
        c_symbol_name.as_ptr(),
    );

    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }

    if symbol_size != std::mem::size_of::<CUdeviceptr>() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }

    let result = cuMemcpyHtoD_v2(
        symbol_address,
        &device_ptr as *const _ as *const std::ffi::c_void,
        symbol_size,
    );

    if result != CUresult::CUDA_SUCCESS {
        return Err(result);
    }

    Ok(())

}
