#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompilerBackend {
    Nvrtc,
    #[cfg(feature = "direct-ptx")]
    DirectPtx {
        version: (u32, u32),
    },
}

#[cfg(test)]
mod tests {
    use super::CompilerBackend;

    #[test]
    fn isolates_nvrtc_math_policy_and_target() {
        assert_eq!(
            CompilerBackend::parse("nvrtc", None)
                .unwrap()
                .cache_namespace(89),
            "cuda-nvrtc-strict-v13-sm89"
        );
        assert_ne!(CompilerBackend::Nvrtc.cache_namespace(89), "cuda");
        assert_ne!(CompilerBackend::Nvrtc.cache_namespace(89), CompilerBackend::Nvrtc.cache_namespace(90));
    }

    #[test]
    fn rejects_unknown_backend() {
        assert!(CompilerBackend::parse("unknown", Some("8.0")).is_err());
    }

    #[cfg(feature = "direct-ptx")]
    #[test]
    fn validates_ptx_version_and_isolates_caches() {
        for version in [
            None,
            Some("8"),
            Some("x.0"),
            Some("8.10"),
            Some("8.0.1"),
            Some("5.0"),
        ] {
            assert!(CompilerBackend::parse("ptx", version).is_err());
        }
        let ptx = CompilerBackend::parse("ptx", Some("8.0")).unwrap();
        assert_ne!(ptx.cache_namespace(89), CompilerBackend::Nvrtc.cache_namespace(89));
        assert_ne!(ptx.cache_namespace(89), ptx.cache_namespace(90));
        assert_ne!(
            ptx.cache_namespace(89),
            CompilerBackend::parse("ptx", Some("8.1"))
                .unwrap()
                .cache_namespace(89)
        );
    }

    #[cfg(not(feature = "direct-ptx"))]
    #[test]
    fn disabled_feature_is_not_a_fallback() {
        assert!(CompilerBackend::parse("ptx", Some("8.0")).is_err());
    }
}

impl CompilerBackend {
    pub fn from_environment() -> Result<Self, String> {
        let backend = match std::env::var("RUDA_CUDA_COMPILER") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return Ok(Self::Nvrtc),
            Err(_) => return Err("RUDA_CUDA_COMPILER must be Unicode".into()),
        };
        Self::parse(&backend, std::env::var("RUDA_PTX_VERSION").ok().as_deref())
    }

    pub fn parse(backend: &str, version: Option<&str>) -> Result<Self, String> {
        match backend {
            "nvrtc" => Ok(Self::Nvrtc),
            "ptx" => {
                #[cfg(feature = "direct-ptx")]
                {
                    let (major, minor) = version
                        .and_then(|s| s.split_once('.'))
                        .ok_or("direct PTX requires RUDA_PTX_VERSION=major.minor")?;
                    let major: u32 = major.parse().map_err(|_| "invalid PTX major version")?;
                    let minor: u32 = minor.parse().map_err(|_| "invalid PTX minor version")?;
                    if major < 6 || minor > 9 {
                        return Err("PTX version must be >= 6.0 with a single minor digit".into());
                    }
                    Ok(Self::DirectPtx {
                        version: (major, minor),
                    })
                }
                #[cfg(not(feature = "direct-ptx"))]
                {
                    let _ = version;
                    Err("RUDA_CUDA_COMPILER=ptx requires the direct-ptx feature".into())
                }
            }
            _ => Err("RUDA_CUDA_COMPILER must be nvrtc or ptx".into()),
        }
    }

    pub fn cache_namespace(self, _sm: u32) -> String {
        match self {
            Self::Nvrtc => format!("cuda-nvrtc-strict-v13-sm{_sm}"),
            #[cfg(feature = "direct-ptx")]
            Self::DirectPtx {
                version: (major, minor),
            } => {
                let revision = ruda_compiler::ptx::PtxCompiler::CACHE_VERSION;
                format!("cuda-direct-ptx-v{revision}-sm{_sm}-ptx{major}_{minor}")
            }
        }
    }
}
