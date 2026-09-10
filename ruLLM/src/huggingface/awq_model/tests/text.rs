use super::*;
use crate::{load_huggingface_awq_qwen2_pipeline, load_huggingface_qwen2_pipeline};
use tokenizers::{
    AddedToken, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
    processors::template::TemplateProcessing,
};

const CHAT_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

fn write_tokenizer(fixture: &Fixture, vocabulary_size: usize) {
    let vocabulary = (0..vocabulary_size)
        .map(|index| {
            let token = match index {
                0 => "[UNK]".into(),
                1 => "hello".into(),
                2 => "world".into(),
                3 => "answer".into(),
                4 => "[BOS]".into(),
                5 => "[EOS]".into(),
                _ => format!("token{index}"),
            };
            (token, index as u32)
        })
        .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.add_special_tokens(&[
        AddedToken::from("[UNK]", true),
        AddedToken::from("[BOS]", true),
        AddedToken::from("[EOS]", true),
    ]);
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[BOS] $A [EOS]")
            .unwrap()
            .special_tokens(vec![("[BOS]", 4), ("[EOS]", 5)])
            .build()
            .unwrap(),
    ));
    tokenizer
        .save(fixture.0.join(TOKENIZER_FILE), false)
        .unwrap();
    std::fs::write(
        fixture.0.join(TOKENIZER_CONFIG_FILE),
        serde_json::to_vec(&json!({"chat_template": CHAT_TEMPLATE})).unwrap(),
    )
    .unwrap();
}

#[test]
fn awq_text_pipeline_preserves_tokenizer_flags_ids_and_plain_chat() {
    let (packed, dense) = model_weights();
    let packed_fixture = Fixture::new(&packed, true, false);
    let dense_fixture = Fixture::new(&dense, false, false);
    write_tokenizer(&packed_fixture, 24);
    write_tokenizer(&dense_fixture, 24);
    let device = CudaDevice::default();
    let pipeline =
        load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(&packed_fixture.0, &device).unwrap();
    let reference =
        load_huggingface_qwen2_pipeline::<TestBackend>(&dense_fixture.0, &device).unwrap();
    assert_eq!(pipeline.chat_template, CHAT_TEMPLATE);
    assert_eq!(pipeline.encode("hello world", false).unwrap(), vec![1, 2]);
    assert_eq!(
        pipeline.encode("hello world", true).unwrap(),
        vec![4, 1, 2, 5]
    );
    for special in [false, true] {
        assert_eq!(
            pipeline.encode("hello world", special).unwrap(),
            reference.encode("hello world", special).unwrap()
        );
        assert_eq!(
            pipeline.decode(&[4, 1, 2, 5], special).unwrap(),
            reference.decode(&[4, 1, 2, 5], special).unwrap()
        );
    }
    assert_eq!(pipeline.decode(&[4, 1, 2, 5], true).unwrap(), "hello world");
    assert_eq!(
        pipeline.decode(&[-1], true).unwrap_err(),
        reference.decode(&[-1], true).unwrap_err()
    );
    let messages = [Qwen2ChatMessage::new(Qwen2ChatRole::User, "hello")];
    for generation_prompt in [false, true] {
        assert_eq!(
            pipeline.render_chat(&messages, generation_prompt).unwrap(),
            reference.render_chat(&messages, generation_prompt).unwrap()
        );
        assert_eq!(
            pipeline.encode_chat(&messages, generation_prompt).unwrap(),
            reference.encode_chat(&messages, generation_prompt).unwrap()
        );
    }
    assert_eq!(
        pipeline.render_chat(&[], true).unwrap_err(),
        reference.render_chat(&[], true).unwrap_err()
    );
}

#[test]
fn awq_text_generation_matches_token_entry_points_and_inherits_generation_eos() {
    let (packed, _) = model_weights();
    let fixture = Fixture::new(&packed, true, false);
    write_tokenizer(&fixture, 24);
    std::fs::write(
        fixture.0.join(GENERATION_CONFIG_FILE),
        br#"{"eos_token_id":[2,7]}"#,
    )
    .unwrap();
    let device = CudaDevice::default();
    let pipeline = load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(&fixture.0, &device).unwrap();
    assert_eq!(pipeline.loaded.default_eos_token_ids, vec![2, 7]);
    let greedy = GreedyGenerationConfig {
        max_new_tokens: 2,
        eos_token_ids: vec![],
    };
    let text = pipeline
        .generate_text("hello world", greedy.clone(), false, true, &device)
        .unwrap();
    let tokens = pipeline
        .loaded
        .generate_tokens(&[1, 2], greedy, &device)
        .unwrap();
    assert_eq!(text.token_ids, tokens.token_ids);
    assert_eq!(text.generated_token_ids, tokens.generated_token_ids);
    assert_eq!(text.stopped_on_eos, tokens.stopped_on_eos);
    assert_eq!(text.text, pipeline.decode(&tokens.token_ids, true).unwrap());
    assert_eq!(
        text.generated_text,
        pipeline.decode(&tokens.generated_token_ids, true).unwrap()
    );
    let sampling = SamplingGenerationConfig {
        max_new_tokens: 2,
        eos_token_ids: vec![],
        sampling: SamplingConfig {
            seed: Some(79),
            ..Default::default()
        },
    };
    let text = pipeline
        .generate_text_sampled("hello world", sampling.clone(), true, false, &device)
        .unwrap();
    let tokens = pipeline
        .loaded
        .generate_tokens_sampled(&[4, 1, 2, 5], sampling, &device)
        .unwrap();
    assert_eq!(text.token_ids, tokens.token_ids);
    assert_eq!(text.generated_token_ids, tokens.generated_token_ids);
    assert_eq!(text.stopped_on_eos, tokens.stopped_on_eos);
    assert_eq!(
        text.text,
        pipeline.decode(&tokens.token_ids, false).unwrap()
    );
    assert_eq!(
        text.generated_text,
        pipeline.decode(&tokens.generated_token_ids, false).unwrap()
    );
}

#[test]
fn awq_text_pipeline_rejects_missing_assets_unsupported_template_and_bad_ids() {
    let (packed, _) = model_weights();
    let fixture = Fixture::new(&packed, true, false);
    let device = CudaDevice::default();
    let missing =
        load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(&fixture.0, &device).unwrap_err();
    assert!(missing.0.contains("tokenizer does not exist"));
    write_tokenizer(&fixture, 24);
    std::fs::write(
        fixture.0.join(TOKENIZER_CONFIG_FILE),
        br#"{"chat_template":"unsupported"}"#,
    )
    .unwrap();
    let template =
        load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(&fixture.0, &device).unwrap_err();
    assert!(template.0.contains("supported plain-message ChatML"));
    write_tokenizer(&fixture, 25);
    let vocabulary =
        load_huggingface_awq_qwen2_pipeline::<CudaRuntime>(&fixture.0, &device).unwrap_err();
    assert!(vocabulary.0.contains("outside model vocabulary"));
}
