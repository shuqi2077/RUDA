use super::*;

impl HipContext {
    /// Compiles a kernel.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        ruda_kernel: Box<dyn RudaTask<HipCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let hash = if let Some(cache) = self.compilation_cache.as_ref() {
            let hash = kernel_id.stable_hash();
            if let Some(entry) = cache.get(&hash) {
                log::trace!("Using compilation cache");
                self.load_compiled_binary(
                    entry.binary.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    kernel_id.ruda_dim,
                    entry.shared_mem_bytes,
                )?;
                return Ok(());
            }
            Some(hash)
        } else {
            None
        };

        validate_ruda_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        // Ruda compilation
        // jitc = just-in-time compiled
        let mut jitc_kernel = ruda_kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            ruda_kernel.address_type(),
        )?;

        self.validate_shared(&jitc_kernel.repr)?;

        if logger.compilation_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jitc_kernel);

        // Create HIP Program
        // SAFETY: Calling HIP RTC FFI to create a program from source. The `CString` ensures
        // the source is null-terminated. The returned `program` handle is valid on success.
        let program = unsafe {
            let source = CString::new(jitc_kernel.source.clone()).unwrap();
            let mut program: ruda_hip_sys::hiprtcProgram = std::ptr::null_mut();

            let status = ruda_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                std::ptr::null(), // program name seems unnecessary
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );

            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to create the program from the source: HIP STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                })?;
            }

            program
        };
        // Compile HIP program
        // options
        let include_path = get_hip_include_path().unwrap();
        let include_option = format!("-I{include_path}");
        let include_option_cstr = CString::new(include_option).unwrap();
        // needed for rocWMMA extension to compile
        let cpp_std_option_cstr = CString::new("--std=c++17").unwrap();
        let optimization_level = CString::new("-O3").unwrap();
        let mut options = vec![
            cpp_std_option_cstr.as_ptr(),
            include_option_cstr.as_ptr(),
            optimization_level.as_ptr(),
        ];
        // SAFETY: `program` is a valid RTC program handle created above. The `options` vector
        // contains valid null-terminated `CString` pointers that outlive this call. On failure,
        // we retrieve and report the compilation log before returning an error.
        unsafe {
            let options_ptr = options.as_mut_ptr();
            let status =
                ruda_hip_sys::hiprtcCompileProgram(program, options.len() as i32, options_ptr);

            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status =
                    ruda_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);

                if status != hiprtcResult_HIPRTC_SUCCESS {
                    Err(CompilationError::Generic {
                        reason: format!(
                            "An error during compilation happened, but we're unable to fetch the error log size. STATUS: {status}"
                        ),
                        backtrace: BackTrace::capture(),
                    })?;
                }

                let mut log_buffer = vec![0; log_size];
                let status = ruda_hip_sys::hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());

                if status != hiprtcResult_HIPRTC_SUCCESS {
                    Err(CompilationError::Generic {
                        reason: format!(
                            "An error during compilation happened, but we're unable to fetch the error log content. STATUS: {status}"
                        ),
                        backtrace: BackTrace::capture(),
                    })?;
                }

                let log = CStr::from_ptr(log_buffer.as_ptr());
                let mut message = "[Compilation Error] ".to_string();
                if log_size > 0 {
                    for line in log.to_string_lossy().split('\n') {
                        if !line.is_empty() {
                            message += format!("\n    {line}").as_str();
                        }
                    }
                } else {
                    message += "\n No compilation logs found!";
                }
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{}", jitc_kernel.source),
                    backtrace: BackTrace::capture(),
                })?;
            }
        };

        // Get HIP compiled code from program
        let mut code_size: usize = 0;
        // SAFETY: `program` was successfully compiled above. `code_size` is a valid mutable
        // pointer to receive the size of the compiled binary.
        unsafe {
            let status = ruda_hip_sys::hiprtcGetCodeSize(program, &mut code_size);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to get the size of the compiled code. STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                })?;
            }
        }
        let mut code = vec![0; code_size];
        // SAFETY: `code` is allocated with `code_size` bytes as reported by `hiprtcGetCodeSize`.
        // `program` is a valid compiled program handle.
        unsafe {
            let status = ruda_hip_sys::hiprtcGetCode(program, code.as_mut_ptr());

            if status != hiprtcResult_HIPRTC_SUCCESS {
                Err(CompilationError::Generic {
                    reason: format!("Unable to get the compiled code. STATUS: {status}"),
                    backtrace: BackTrace::capture(),
                })?;
            }
        }

        let repr = jitc_kernel.repr.unwrap();

        if let Some(cache) = self.compilation_cache.as_mut() {
            cache
                .insert(
                    hash.unwrap(),
                    CompilationCacheEntry {
                        entrypoint_name: jitc_kernel.entrypoint_name.clone(),
                        shared_mem_bytes: repr.shared_memory_size(),
                        binary: code.clone(),
                    },
                )
                .unwrap();
        }

        self.load_compiled_binary(
            code,
            kernel_id.clone(),
            jitc_kernel.entrypoint_name,
            jitc_kernel.ruda_dim,
            repr.shared_memory_size(),
        )?;
        Ok(())
    }

    fn load_compiled_binary(
        &mut self,
        code: Vec<i8>,
        kernel_id: KernelId,
        entrypoint_name: String,
        ruda_dim: RudaDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name.clone()).unwrap();

        // Create the HIP module
        let mut module: ruda_hip_sys::hipModule_t = std::ptr::null_mut();
        // SAFETY: `code` contains a valid compiled binary obtained from `hiprtcGetCode`.
        // `module` receives the loaded module handle on success.
        unsafe {
            let codeptr = code.as_ptr();
            let status = ruda_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                return Err(CompilationError::Generic {
                    reason: format!("Unable to load the compiled module. STATUS: {status}"),
                    backtrace: BackTrace::capture(),
                });
            }
        }
        // Retrieve the HIP module function
        let mut func: ruda_hip_sys::hipFunction_t = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded module from `hipModuleLoadData` above.
        // `func_name` is a null-terminated `CString` matching the kernel entry point.
        unsafe {
            let status =
                ruda_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            if status != hiprtcResult_HIPRTC_SUCCESS {
                return Err(CompilationError::Generic {
                    reason: format!(
                        "Unable to load the function in the compiled module. STATUS: {status}"
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        // register module
        self.module_names.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                ruda_dim,
                shared_mem_bytes,
            },
        );

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<HipComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
