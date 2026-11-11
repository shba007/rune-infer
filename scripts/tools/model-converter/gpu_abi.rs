//! Validation-only LiteRT 2.3.0 Windows WebGPU ABI. Never linked into Infer.
//! ABI source: the checksum-pinned official v2.3.0 C++ SDK's litert/c headers.
#[cfg(all(windows, target_arch = "x86_64"))]
#[allow(non_snake_case)]
mod windows {
    use std::{
        ffi::{CString, c_char, c_void},
        io::{Read, Write},
        path::Path,
        ptr,
    };
    type Handle = *mut c_void;
    type Status = i32;
    #[repr(C)]
    struct Any {
        kind: i32,
        value: u64,
    }
    #[repr(C)]
    struct EnvOption {
        tag: i32,
        value: Any,
    }
    const _: () = assert!(std::mem::size_of::<EnvOption>() == 24);
    unsafe extern "C" fn borrowed_payload(_: Handle) {}
    #[repr(C)]
    #[derive(Default)]
    struct Layout {
        flags: u32,
        dimensions: [i32; 8],
        strides: [u32; 8],
    }
    #[repr(C)]
    #[derive(Default)]
    struct TensorType {
        element_type: i32,
        layout: Layout,
    }
    const _: () = assert!(std::mem::size_of::<TensorType>() == 72);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(path: *const u16, file: Handle, flags: u32) -> Handle;
        fn GetProcAddress(module: Handle, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: Handle) -> i32;
    }
    struct Library(Handle);
    impl Drop for Library {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.0);
            }
        }
    }
    impl Library {
        unsafe fn symbol<T: Copy>(&self, name: &'static [u8]) -> anyhow::Result<T> {
            let address = unsafe { GetProcAddress(self.0, name.as_ptr().cast()) };
            anyhow::ensure!(
                !address.is_null(),
                "Missing LiteRT symbol {}",
                String::from_utf8_lossy(name)
            );
            // Each use below names the function signature from the pinned C ABI.
            Ok(unsafe { std::mem::transmute_copy(&address) })
        }
    }
    fn check(status: Status) -> anyhow::Result<()> {
        anyhow::ensure!(status == 0, "LiteRT status {status}");
        Ok(())
    }
    // Owned native objects are destroyed in reverse order, before unloading the DLL.
    struct Owned {
        handle: Handle,
        destroy: unsafe extern "C" fn(Handle),
    }
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe {
                (self.destroy)(self.handle);
            }
        }
    }

    pub fn run(runtime: &Path, model_path: &Path) -> anyhow::Result<()> {
        use std::os::windows::ffi::OsStrExt;
        super::super::verify_artifacts(runtime, model_path)?;
        let path: Vec<_> = std::fs::canonicalize(runtime)?
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        // Search only this DLL's directory and System32, never CWD or PATH.
        let library = Library(unsafe { LoadLibraryExW(path.as_ptr(), ptr::null_mut(), 0x900) });
        anyhow::ensure!(
            !library.0.is_null(),
            "Cannot load LiteRT runtime: {}",
            std::io::Error::last_os_error()
        );
        macro_rules! api {
            ($name:ident, $ty:ty) => {
                let $name: $ty =
                    unsafe { library.symbol(concat!(stringify!($name), "\0").as_bytes())? };
            };
        }
        api!(
            LiteRtCreateEnvironment,
            unsafe extern "C" fn(i32, Handle, *mut Handle) -> Status
        );
        api!(
            LiteRtCreateOpaqueOptions,
            unsafe extern "C" fn(
                *const c_char,
                Handle,
                unsafe extern "C" fn(Handle),
                *mut Handle,
            ) -> Status
        );
        api!(
            LiteRtAddOpaqueOptions,
            unsafe extern "C" fn(Handle, Handle) -> Status
        );
        api!(
            LiteRtCompiledModelIsNonCpuFullyAccelerated,
            unsafe extern "C" fn(Handle, *mut bool) -> Status
        );
        api!(
            LiteRtCreateOptions,
            unsafe extern "C" fn(*mut Handle) -> Status
        );
        api!(
            LiteRtSetOptionsHardwareAccelerators,
            unsafe extern "C" fn(Handle, i32) -> Status
        );
        api!(
            LiteRtCreateModelFromFile,
            unsafe extern "C" fn(Handle, *const c_char, *mut Handle) -> Status
        );
        api!(
            LiteRtGetModelSubgraph,
            unsafe extern "C" fn(Handle, usize, *mut Handle) -> Status
        );
        api!(
            LiteRtCreateCompiledModel,
            unsafe extern "C" fn(Handle, Handle, Handle, *mut Handle) -> Status
        );
        api!(
            LiteRtRunCompiledModel,
            unsafe extern "C" fn(Handle, usize, usize, *mut Handle, usize, *mut Handle) -> Status
        );
        api!(
            LiteRtGetRankedTensorType,
            unsafe extern "C" fn(Handle, *mut TensorType) -> Status
        );
        api!(
            LiteRtCreateManagedTensorBuffer,
            unsafe extern "C" fn(Handle, i32, *const TensorType, usize, *mut Handle) -> Status
        );
        api!(
            LiteRtGetTensorBufferRequirementsBufferSize,
            unsafe extern "C" fn(Handle, *mut usize) -> Status
        );
        api!(
            LiteRtLockTensorBuffer,
            unsafe extern "C" fn(Handle, *mut Handle, i32) -> Status
        );
        api!(
            LiteRtUnlockTensorBuffer,
            unsafe extern "C" fn(Handle) -> Status
        );
        macro_rules! owned {
            ($creator:expr, $destroy:literal) => {{
                let mut handle = ptr::null_mut();
                check($creator(&mut handle))?;
                Owned {
                    handle,
                    destroy: library.symbol(concat!($destroy, "\0").as_bytes())?,
                }
            }};
        }
        #[allow(non_snake_case)]
        unsafe fn work(
            library: &Library,
            tensor_type: unsafe extern "C" fn(Handle, *mut TensorType) -> Status,
            create_buffer: unsafe extern "C" fn(
                Handle,
                i32,
                *const TensorType,
                usize,
                *mut Handle,
            ) -> Status,
            buffer_size: unsafe extern "C" fn(Handle, *mut usize) -> Status,
            env: Handle,
            subgraph: Handle,
            compiled: Handle,
            input: bool,
        ) -> anyhow::Result<Owned> {
            let tensor_name: &[u8] = if input {
                b"LiteRtGetSubgraphInput\0"
            } else {
                b"LiteRtGetSubgraphOutput\0"
            };
            let getter: unsafe extern "C" fn(Handle, usize, *mut Handle) -> Status =
                unsafe { library.symbol(tensor_name)? };
            let count_name: &[u8] = if input {
                b"LiteRtGetNumSubgraphInputs\0"
            } else {
                b"LiteRtGetNumSubgraphOutputs\0"
            };
            let count: unsafe extern "C" fn(Handle, *mut usize) -> Status =
                unsafe { library.symbol(count_name)? };
            let mut n = 0;
            check(unsafe { count(subgraph, &mut n) })?;
            anyhow::ensure!(n == 1, "Depth model must have exactly one input and output");
            let mut tensor = ptr::null_mut();
            check(unsafe { getter(subgraph, 0, &mut tensor) })?;
            let mut ty = TensorType::default();
            check(unsafe { tensor_type(tensor, &mut ty) })?;
            let probe_shape = super::super::probe_shape()?;
            let expected: &[i32] = if !input && probe_shape.is_some() {
                probe_shape.as_deref().unwrap()
            } else if std::env::var_os("RUNE_LITERT_VALIDATE_NHWC").is_some() {
                if input {
                    &[1, 518, 686, 3]
                } else {
                    &[1, 518, 686, 1]
                }
            } else if input {
                &[1, 3, 518, 686]
            } else {
                &[1, 518, 686]
            };
            anyhow::ensure!(
                ty.element_type == 1
                    && ty.layout.flags == expected.len() as u32
                    && &ty.layout.dimensions[..expected.len()] == expected,
                "Unexpected depth tensor layout or type"
            );
            let req_name: &[u8] = if input {
                b"LiteRtGetCompiledModelInputBufferRequirements\0"
            } else {
                b"LiteRtGetCompiledModelOutputBufferRequirements\0"
            };
            let req: unsafe extern "C" fn(Handle, usize, usize, *mut Handle) -> Status =
                unsafe { library.symbol(req_name)? };
            let mut requirements = ptr::null_mut();
            check(unsafe { req(compiled, 0, 0, &mut requirements) })?;
            let mut size = 0;
            check(unsafe { buffer_size(requirements, &mut size) })?;
            let logical = expected.iter().product::<i32>() as usize * 4;
            anyhow::ensure!(
                (logical..=64 * 1024 * 1024).contains(&size),
                "Invalid depth buffer requirements"
            );
            let mut handle = ptr::null_mut();
            let count_types: unsafe extern "C" fn(Handle, *mut i32) -> Status = unsafe {
                library.symbol(b"LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes\0")?
            };
            let get_type: unsafe extern "C" fn(Handle, i32, *mut i32) -> Status = unsafe {
                library.symbol(b"LiteRtGetTensorBufferRequirementsSupportedTensorBufferType\0")?
            };
            let mut count = 0;
            check(unsafe { count_types(requirements, &mut count) })?;
            anyhow::ensure!((1..=32).contains(&count), "Invalid buffer type count");
            let mut types = Vec::new();
            for index in 0..count {
                let mut kind = 0;
                check(unsafe { get_type(requirements, index, &mut kind) })?;
                types.push(kind);
            }
            eprintln!("input={input} supported_buffers={types:?} bytes={size}");
            if !input && probe_shape.is_none() {
                anyhow::ensure!(
                    types.iter().any(|kind| (20..=29).contains(kind)),
                    "Validation requires a WebGPU output buffer; CPU-only execution is not GPU proof"
                );
            }
            let kind = if types.contains(&1) { 1 } else { types[0] };
            check(unsafe { create_buffer(env, kind, &ty, size, &mut handle) })?;
            Ok(Owned {
                handle,
                destroy: unsafe { library.symbol(b"LiteRtDestroyTensorBuffer\0")? },
            })
        }
        // The entire ABI boundary runs inside this subprocess, never the gateway.
        #[allow(non_snake_case)]
        unsafe {
            let directory = CString::new(
                std::fs::canonicalize(runtime)?
                    .parent()
                    .unwrap()
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Runtime path is not UTF-8"))?,
            )?;
            let mut env_options = [EnvOption {
                tag: 22,
                value: Any {
                    kind: 8,
                    value: directory.as_ptr() as u64,
                },
            }];
            let env = owned!(
                |out| LiteRtCreateEnvironment(1, env_options.as_mut_ptr().cast(), out),
                "LiteRtDestroyEnvironment"
            );
            // Upstream LrtGetOpaqueGpuOptionsData serializes these enum values as
            // a NUL-terminated TOML string. Keep it alive beyond options/model.
            let payload = CString::new(if std::env::var_os("RUNE_LITERT_VALIDATE_IR").is_some() {
                "precision = 2\nbackend = 2\nuse_ir_model = true\n"
            } else {
                "precision = 2\nbackend = 2\n"
            })?;
            let options = owned!(|out| LiteRtCreateOptions(out), "LiteRtDestroyOptions");
            let opaque = owned!(
                |out| LiteRtCreateOpaqueOptions(
                    c"gpu_options".as_ptr(),
                    payload.as_ptr().cast_mut().cast(),
                    borrowed_payload,
                    out
                ),
                "LiteRtDestroyOpaqueOptions"
            );
            check(LiteRtAddOpaqueOptions(options.handle, opaque.handle))?;
            std::mem::forget(opaque); // Options now owns this object.
            // CPU must be included for the 24 unsupported shape/index ops.
            // GPU delegation is checked separately from full-GPU delegation.
            check(LiteRtSetOptionsHardwareAccelerators(options.handle, 3))?;
            let filename = CString::new(
                std::fs::canonicalize(model_path)?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Model path is not UTF-8"))?,
            )?;
            let model = owned!(
                |out| LiteRtCreateModelFromFile(env.handle, filename.as_ptr(), out),
                "LiteRtDestroyModel"
            );
            let started = std::time::Instant::now();
            let compiled = owned!(
                |out| LiteRtCreateCompiledModel(env.handle, model.handle, options.handle, out),
                "LiteRtDestroyCompiledModel"
            );
            let mut fully_gpu = false;
            check(LiteRtCompiledModelIsNonCpuFullyAccelerated(
                compiled.handle,
                &mut fully_gpu,
            ))?;
            eprintln!(
                "webgpu_fp32 non_cpu_fully_accelerated={fully_gpu} compile_seconds={}",
                started.elapsed().as_secs_f64()
            );
            // Record partial delegation honestly; GPU-unsupported shape/index
            // operations may execute on CPU even with GPU selected.
            let mut subgraph = ptr::null_mut();
            check(LiteRtGetModelSubgraph(model.handle, 0, &mut subgraph))?;
            let input = work(
                &library,
                LiteRtGetRankedTensorType,
                LiteRtCreateManagedTensorBuffer,
                LiteRtGetTensorBufferRequirementsBufferSize,
                env.handle,
                subgraph,
                compiled.handle,
                true,
            )?;
            let output = work(
                &library,
                LiteRtGetRankedTensorType,
                LiteRtCreateManagedTensorBuffer,
                LiteRtGetTensorBufferRequirementsBufferSize,
                env.handle,
                subgraph,
                compiled.handle,
                false,
            )?;
            let mut data = vec![0u8; super::super::PIXELS * 3 * 4];
            std::io::stdin().read_exact(&mut data)?;
            let mut address = ptr::null_mut();
            check(LiteRtLockTensorBuffer(input.handle, &mut address, 1))?;
            anyhow::ensure!(!address.is_null(), "Null LiteRT input buffer");
            ptr::copy_nonoverlapping(data.as_ptr(), address.cast::<u8>(), data.len());
            check(LiteRtUnlockTensorBuffer(input.handle))?;
            let started = std::time::Instant::now();
            check(LiteRtRunCompiledModel(
                compiled.handle,
                0,
                1,
                &mut [input.handle][0],
                1,
                &mut [output.handle][0],
            ))?;
            eprintln!("run_seconds={}", started.elapsed().as_secs_f64());
            check(LiteRtLockTensorBuffer(output.handle, &mut address, 0))?;
            anyhow::ensure!(!address.is_null(), "Null LiteRT output buffer");
            let output_elements = super::super::probe_shape()?
                .map(|shape| shape.iter().map(|value| *value as usize).product())
                .unwrap_or(super::super::PIXELS);
            let bytes = std::slice::from_raw_parts(address.cast::<u8>(), output_elements * 4);
            let result = std::io::stdout().write_all(bytes);
            check(LiteRtUnlockTensorBuffer(output.handle))?;
            result?;
        }
        Ok(())
    }
}

pub fn run(runtime: &std::path::Path, model: &std::path::Path) -> anyhow::Result<()> {
    #[cfg(all(windows, target_arch = "x86_64"))]
    {
        windows::run(runtime, model)
    }
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    {
        let _ = (runtime, model);
        anyhow::bail!("LiteRT depth worker currently requires Windows x64 CPU")
    }
}
