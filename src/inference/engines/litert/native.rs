//! LiteRT 2.3.0 C ABI, used only by the isolated worker subprocess.
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
            let expected: &[i32] = if input {
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
            check(unsafe { create_buffer(env, 1, &ty, size, &mut handle) })?;
            Ok(Owned {
                handle,
                destroy: unsafe { library.symbol(b"LiteRtDestroyTensorBuffer\0")? },
            })
        }
        // The entire ABI boundary runs inside this subprocess, never the gateway.
        #[allow(non_snake_case)]
        unsafe {
            let env = owned!(
                |out| LiteRtCreateEnvironment(0, ptr::null_mut(), out),
                "LiteRtDestroyEnvironment"
            );
            let options = owned!(|out| LiteRtCreateOptions(out), "LiteRtDestroyOptions");
            check(LiteRtSetOptionsHardwareAccelerators(options.handle, 1))?;
            let filename = CString::new(
                std::fs::canonicalize(model_path)?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Model path is not UTF-8"))?,
            )?;
            let model = owned!(
                |out| LiteRtCreateModelFromFile(env.handle, filename.as_ptr(), out),
                "LiteRtDestroyModel"
            );
            let compiled = owned!(
                |out| LiteRtCreateCompiledModel(env.handle, model.handle, options.handle, out),
                "LiteRtDestroyCompiledModel"
            );
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
            check(LiteRtRunCompiledModel(
                compiled.handle,
                0,
                1,
                &mut [input.handle][0],
                1,
                &mut [output.handle][0],
            ))?;
            check(LiteRtLockTensorBuffer(output.handle, &mut address, 0))?;
            anyhow::ensure!(!address.is_null(), "Null LiteRT output buffer");
            let bytes = std::slice::from_raw_parts(address.cast::<u8>(), super::super::PIXELS * 4);
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
