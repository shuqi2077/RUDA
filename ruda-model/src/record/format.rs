use alloc::string::String;

pub(super) fn record_type_name(name: &str) -> String {
    String::from(name)
}

#[cfg(test)]
mod tests {
    use super::record_type_name;
    use crate::record::{BinBytesRecorder, RudaRecord, FullPrecisionSettings};
    use core::any::type_name;

    #[test]
    fn preserves_builtin_recorder_metadata() {
        type R = BinBytesRecorder<FullPrecisionSettings>;
        let expected = "ruda_model::record::memory::BinBytesRecorder<ruda_model::record::settings::FullPrecisionSettings, alloc::vec::Vec<u8>>";
        assert_eq!(record_type_name(type_name::<R>()), expected);
        let record = RudaRecord::<_, crate::TestBackend>::new::<R>(());
        assert_eq!(record.metadata.format, expected);
        assert_eq!(record.metadata.version, "0.21.0");
    }

    #[test]
    fn preserves_custom_recorder_paths_and_maps_native_generic_arguments() {
        assert_eq!(
            record_type_name("client::Recorder<ruda_model::record::settings::FullPrecisionSettings>"),
            "client::Recorder<ruda_model::record::settings::FullPrecisionSettings>"
        );
        for name in [
            "client::Recorder<client::Settings>",
            "client::ruda_model::Recorder<client::ruda_model::Settings>",
            "my_ruda_model::Recorder",
            "client_ruda_model::record::memory::BinBytesRecorder",
        ] {
            assert_eq!(record_type_name(name), name);
        }
    }

    #[test]
    fn preserves_compound_type_structure() {
        assert_eq!(
            record_type_name("client::Recorder<(ruda_model::A, &ruda_model::B, [ruda_model::C; 2])>"),
            "client::Recorder<(ruda_model::A, &ruda_model::B, [ruda_model::C; 2])>"
        );
        assert_eq!(
            record_type_name("<ruda_model::A as client::Trait<ruda_model::B>>::Item"),
            "<ruda_model::A as client::Trait<ruda_model::B>>::Item"
        );
    }

    #[test]
    fn preserves_neural_network_generic_type_identities() {
        assert_eq!(
            record_type_name("client::Recorder<ruda_nn::modules::linear::Linear<client::Backend>, ruda_model::record::settings::FullPrecisionSettings>"),
            "client::Recorder<ruda_nn::modules::linear::Linear<client::Backend>, ruda_model::record::settings::FullPrecisionSettings>"
        );
        for name in [
            "client::ruda_nn::Recorder<client::ruda_nn::Settings>",
            "client_ruda_nn::Recorder",
            "ruda_unknown::Recorder",
        ] {
            assert_eq!(record_type_name(name), name);
        }
    }

    #[test]
    fn preserves_optimizer_generic_type_identities() {
        assert_eq!(
            record_type_name("client::Recorder<ruda_optim::optim::adam::AdamState<client::Backend, 2>, ruda_model::record::settings::FullPrecisionSettings>"),
            "client::Recorder<ruda_optim::optim::adam::AdamState<client::Backend, 2>, ruda_model::record::settings::FullPrecisionSettings>"
        );
        for name in [
            "client::ruda_optim::Recorder<client::ruda_optim::Settings>",
            "client_ruda_optim::Recorder",
        ] {
            assert_eq!(record_type_name(name), name);
        }
    }

    #[test]
    fn preserves_store_generic_type_identities() {
        assert_eq!(
            record_type_name("client::Recorder<ruda_store::adapter::HalfPrecisionAdapter>"),
            "client::Recorder<ruda_store::adapter::HalfPrecisionAdapter>"
        );
        for name in [
            "client::ruda_store::Recorder<client::ruda_store::Settings>",
            "client_ruda_store::Recorder",
        ] {
            assert_eq!(record_type_name(name), name);
        }
    }

    #[test]
    #[cfg(feature = "std")]
    fn preserves_file_recorder_and_precision_identities() {
        use crate::record::*;
        for (actual, expected) in [
            (type_name::<BinFileRecorder<FullPrecisionSettings>>(), "ruda_model::record::file::BinFileRecorder<ruda_model::record::settings::FullPrecisionSettings>"),
            (type_name::<BinGzFileRecorder<HalfPrecisionSettings>>(), "ruda_model::record::file::BinGzFileRecorder<ruda_model::record::settings::HalfPrecisionSettings>"),
            (type_name::<JsonGzFileRecorder<DoublePrecisionSettings>>(), "ruda_model::record::file::JsonGzFileRecorder<ruda_model::record::settings::DoublePrecisionSettings>"),
            (type_name::<PrettyJsonFileRecorder<FullPrecisionSettings>>(), "ruda_model::record::file::PrettyJsonFileRecorder<ruda_model::record::settings::FullPrecisionSettings>"),
            (type_name::<NamedMpkGzFileRecorder<HalfPrecisionSettings>>(), "ruda_model::record::file::NamedMpkGzFileRecorder<ruda_model::record::settings::HalfPrecisionSettings>"),
            (type_name::<DefaultFileRecorder<DoublePrecisionSettings>>(), "ruda_model::record::file::NamedMpkFileRecorder<ruda_model::record::settings::DoublePrecisionSettings>"),
            (type_name::<NamedMpkBytesRecorder<FullPrecisionSettings>>(), "ruda_model::record::memory::NamedMpkBytesRecorder<ruda_model::record::settings::FullPrecisionSettings>"),
            (type_name::<NoStdInferenceRecorder>(), "ruda_model::record::memory::BinBytesRecorder<ruda_model::record::settings::FullPrecisionSettings, &[u8]>"),
        ] {
            assert_eq!(record_type_name(actual), expected);
        }
    }
}
