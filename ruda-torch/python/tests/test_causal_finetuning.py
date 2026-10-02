"""CPU tests of production generic SFT; not large-model GPU acceptance."""
import copy
import argparse
import importlib
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from architecture_test_utils import NAME

cft = importlib.import_module(NAME + '.causal_finetuning')
ft = importlib.import_module(NAME + '.finetuning')
torch.set_num_threads(1)


@pytest.mark.parametrize('kind', ['dense', 'nf4', 'lora', 'qlora'])
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('shift', [False, True])
def test_chunked_loss_and_all_gradients_match_full_vocabulary(kind, recompute, shift):
    torch.manual_seed(102)
    head = nn.Linear(7, 19)
    if kind in ('nf4', 'qlora'):
        head = ft.NF4Linear.from_linear(head, block_size=8, tile_rows=3)
    if kind in ('lora', 'qlora'):
        head = ft.LoRALinear(head, rank=3, alpha=6.)
        with torch.no_grad():
            head.lora_B.normal_(std=.05)
    ref_head = copy.deepcopy(head)
    x = torch.randn(2, 6, 7, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    labels = torch.randint(0, 19, (2, 6)); labels[0, :2] = -100; labels[1, -2:] = -100
    # A complete-logit independent loss oracle; same frozen NF4 codebook.
    logits = ref_head(xr[:, :-1] if shift else xr)
    expected = F.cross_entropy(logits.reshape(-1, 19), (labels[:, 1:] if shift else labels).reshape(-1))
    actual = cft.chunked_lm_cross_entropy(x, head, labels, shift=shift,
                                        token_chunk_size=3, recompute=recompute)
    actual.backward(); expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad, rtol=3e-5, atol=2e-7)
    for p, q in zip(head.parameters(), ref_head.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=4e-5, atol=3e-7)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_all_ignored_loss_is_connected_zero(dtype):
    head = nn.Linear(5, 11).to(dtype)
    x = torch.randn(2, 4, 5, dtype=dtype, requires_grad=True)
    loss = cft.chunked_lm_cross_entropy(x, head, torch.full((2, 4), -100), token_chunk_size=2)
    loss.backward()
    assert loss.item() == 0 and torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(head.weight.grad, torch.zeros_like(head.weight))


def test_chunked_projection_never_retains_full_token_vocabulary():
    head = nn.Linear(5, 31)
    projected_rows = []
    head.register_forward_hook(lambda module, args, result: projected_rows.append(result.shape[0]))
    x = torch.randn(3, 7, 5, requires_grad=True)
    labels = torch.randint(0, 31, (3, 7))
    cft.chunked_lm_cross_entropy(x, head, labels, token_chunk_size=4).backward()
    assert len(projected_rows) >= 10 and max(projected_rows) <= 4


def test_collator_separates_attention_and_supervision_and_refuses_implicit_truncation():
    collate = cft.SFTCollator(max_length=8, pad_token_id=0)
    batch = collate([{'input_ids': [2, 3, 4], 'labels': [-100, -100, 4]},
                     {'input_ids': [5, 6], 'labels': [-100, 6]}])
    assert batch['labels'].tolist() == [[-100, -100, 4], [-100, 6, -100]]
    assert batch['attention_mask'].tolist() == [[True, True, True], [True, True, False]]
    with pytest.raises(ValueError, match='max_length'):
        collate([{'input_ids': [1] * 9, 'labels': [1] * 9}])
    with pytest.raises(ValueError, match='pad_token'):
        cft.SFTCollator(max_length=8)


def test_chat_supervision_uses_tokenizer_declared_spans():
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['return_assistant_tokens_mask'] and not kwargs['add_generation_prompt']
            return {'input_ids': [1, 2, 3, 4], 'assistant_masks': [0, 0, 1, 1]}
    batch = cft.SFTCollator(Tokenizer(), max_length=8)([{'messages': [{'role': 'assistant', 'content': 'text'}]}])
    assert batch['labels'].tolist() == [[-100, -100, 3, 4]]
    class Missing(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {'input_ids': [1, 2]}
    with pytest.raises(ValueError, match='declare assistant'):
        cft.SFTCollator(Missing(), max_length=8)([{'messages': []}])


class Backbone(nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.embedding = nn.Embedding(23, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, input_ids, attention_mask):
        return self.block(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)


def make_model(dropout=0., checkpoint=False):
    base = Backbone(dropout)
    ft.inject_lora(base, target_modules=['block.0'], rank=3, alpha=3.)
    head = nn.Linear(8, 23)
    head.requires_grad_(False)
    return cft.CausalLMFinetuner(base, head, token_chunk_size=3, activation_checkpointing=checkpoint)


def batch(length=6):
    labels = list(range(1, length + 1)); labels[0] = -100
    return cft.SFTCollator(max_length=8, pad_token_id=0)([{'input_ids': list(range(1, length + 1)), 'labels': labels}])


def test_token_weighted_accumulation_matches_single_loss():
    torch.manual_seed(109)
    model = make_model(); reference = copy.deepcopy(model)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
    ropt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=.05)
    trainer = cft.SFTTrainer(model, opt, base_id='unit', run_config={'data': 'unit'})
    batches = [batch(3), batch(7)]
    metrics = trainer.train_step(batches)
    loss = sum(reference(**b, reduction='sum') for b in batches) / 8
    loss.backward(); ropt.step()
    assert metrics['supervised_tokens'] == 8 and trainer.cursor == 2
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize('quantized', [False, True])
def test_sft_reuses_existing_aot_compiler_without_changing_checkpoint_names(tmp_path, quantized):
    compiler = importlib.import_module(NAME + '.compiler')
    torch.manual_seed(119)
    model = make_model(checkpoint=True)
    if quantized:
        layer = model.backbone.block[0]
        layer.base = ft.NF4Linear.from_linear(layer.base, block_size=8, tile_rows=3)
    reference = copy.deepcopy(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005, foreach=False)
    ropt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=.005, foreach=False)
    wrapped = compiler.compile(model, device_type='cpu', native='off', fullgraph=True)
    trainer = cft.SFTTrainer(wrapped, opt, base_id='test-compiled-base', run_config={'compiled': True})
    other = cft.SFTTrainer(reference, ropt, base_id='test-compiled-base', run_config={'compiled': True})
    try:
        for _ in range(2):
            got, want = trainer.train_step([batch()]), other.train_step([batch()])
            assert got['loss'] == pytest.approx(want['loss'], rel=2e-6)
            for p, q in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(p, q, rtol=3e-5, atol=3e-7)
        phases = {graph['phase'] for graph in wrapped.info['graphs']}
        assert {'forward', 'backward'} <= phases
        path = trainer.save(tmp_path)
        state = torch.load(path, weights_only=True)
        assert all(not name.startswith('_original.') for group in state['optimizer_layout'] for name in group)
        other.resume(path)
    finally:
        wrapped.close()


def test_checkpoint_actual_continuation_restores_rng_optimizer_and_cursor(tmp_path):
    torch.manual_seed(110)
    model = make_model(.2, checkpoint=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, foreach=False)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.8)
    trainer = cft.SFTTrainer(model, opt, base_id='unit-frozen-checkpoint', run_config={'dataset': 'exact'}, scheduler=schedule)
    trainer.train_step([batch()]); path = trainer.save(tmp_path)
    state = torch.load(path, weights_only=True)
    assert all('lora_' in key or key in ('format', 'version', 'layers') for key in state['adapter'])
    expected = trainer.train_step([batch()])
    expected_parameters = [p.detach().clone() for p in model.parameters()]
    restored = copy.deepcopy(model)
    ropt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.7, foreach=False)
    rsched = torch.optim.lr_scheduler.StepLR(ropt, step_size=1, gamma=.8)
    second = cft.SFTTrainer(restored, ropt, base_id='unit-frozen-checkpoint', run_config={'dataset': 'exact'}, scheduler=rsched)
    assert second.resume(path) == 1 and second.cursor == 1
    actual = second.train_step([batch()])
    assert actual['loss'] == expected['loss'] and second.cursor == trainer.cursor == 2
    for p, q in zip(restored.parameters(), expected_parameters):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert ropt.param_groups[0]['lr'] == opt.param_groups[0]['lr']
    second.save(tmp_path)
    assert torch.load(tmp_path / 'previous.pt', weights_only=True)['step'] == 1
    second.write_progress(tmp_path, actual, total_steps=4)
    assert json.loads((tmp_path / 'progress.json').read_text())['step'] == 2
    second.run_config['dataset'] = 'changed'
    with pytest.raises(ValueError, match='snapshot'):
        second.resume(path)


def test_explicit_checkpoint_paths_preserve_keys_and_stochastic_gradients():
    torch.manual_seed(111)
    base = Backbone(.2); ref = copy.deepcopy(base)
    keys = list(base.state_dict())
    cft.activation_checkpoint_modules(base, ['block'])
    assert list(base.state_dict()) == keys
    inputs = batch()
    rng = torch.get_rng_state()
    actual = base(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']); actual.square().mean().backward()
    torch.set_rng_state(rng)
    expected = ref(inputs['input_ids'], inputs['attention_mask']); expected.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, q in zip(base.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match='already'):
        cft.activation_checkpoint_modules(base, ['block'])


def test_stream_loader_preserves_buffer_dtype_and_explicit_parameters(tmp_path):
    from safetensors.torch import save_file
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 7, device='meta')
            self.precise = nn.Parameter(torch.empty(7, device='meta'))
            self.register_buffer('rotary', torch.tensor([.1, .2], dtype=torch.float32), persistent=False)
            self.register_buffer('counts', torch.tensor([1, 2], dtype=torch.int64))
    save_file({'linear.weight': torch.randn(7, 5), 'linear.bias': torch.randn(7),
               'precise': torch.randn(7), 'counts': torch.tensor([2, 3])}, tmp_path / 'model.safetensors')
    model = Base()
    ft.load_nf4_safetensors(model, tmp_path, target_modules=['linear'], device='cpu', dtype=torch.bfloat16,
                            parameter_dtypes={'precise': torch.float32})
    assert model.precise.dtype == model.rotary.dtype == torch.float32
    assert model.counts.dtype == torch.int64 and model.linear.bias.dtype == torch.bfloat16
    with pytest.raises(ValueError, match='dtype-preserved'):
        ft.load_nf4_safetensors(Base(), tmp_path, target_modules=['linear'], device='cpu',
                                parameter_dtypes={'linear.weight': torch.float32})


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('family', ['llama', 'gpt_neox'])
def test_real_local_hf_checkpoint_streaming_sft_and_adapter_resume(tmp_path, dtype, family):
    from transformers import LlamaConfig, LlamaForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(112)
    if family == 'llama':
        config = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                             tie_word_embeddings=True, attention_dropout=0.)
        original = LlamaForCausalLM(config)
        backbone, head = 'model', 'lm_head'
        targets = ['model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.v_proj']
    else:
        config = GPTNeoXConfig(vocab_size=23, hidden_size=16, intermediate_size=24,
                               num_hidden_layers=1, num_attention_heads=2, rotary_pct=.5,
                               tie_word_embeddings=True, attention_dropout=0., hidden_dropout=0.)
        original = GPTNeoXForCausalLM(config)
        backbone, head = 'gpt_neox', 'embed_out'
        targets = ['gpt_neox.layers.0.attention.query_key_value', 'gpt_neox.layers.0.attention.dense']
    directory = tmp_path / 'base'; original.save_pretrained(directory, safe_serialization=True)
    loaded = cft.load_hf_nf4_model(directory, target_modules=targets, device='cpu', dtype=dtype,
                                   rank=2, alpha=2., block_size=8,
                                   model_kwargs={'attn_implementation': 'eager'})
    assert loaded.get_output_embeddings().weight is loaded.get_input_embeddings().weight
    assert isinstance(loaded.get_submodule(targets[0]), ft.LoRALinear)
    assert loaded.get_submodule(backbone).rotary_emb.inv_freq.dtype == torch.float32
    model = cft.CausalLMFinetuner(loaded.get_submodule(backbone), loaded.get_submodule(head), token_chunk_size=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005, foreach=False)
    trainer = cft.SFTTrainer(model, optimizer, base_id='local-unit-llama', run_config={'source': 'test'})
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    metrics = trainer.train_step([batch()])
    assert metrics['loss'] > 0
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters() if not p.requires_grad)
    path = trainer.save(tmp_path / 'checkpoint'); trainer.resume(path)
    trainer.train_step([batch()]); assert trainer.step == 2


@pytest.mark.parametrize('compiled', [False, True])
def test_generic_cli_consumes_real_jsonl_and_resumes_without_repeating_batches(tmp_path, compiled):
    import importlib.util
    from pathlib import Path
    from transformers import LlamaConfig, LlamaForCausalLM
    source = Path(__file__).resolve().parents[1] / 'examples/finetune_causal_lm.py'
    spec = importlib.util.spec_from_file_location('generic_sft_example', source)
    example = importlib.util.module_from_spec(spec); spec.loader.exec_module(example)
    torch.manual_seed(114)
    config = LlamaConfig(vocab_size=23, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         attention_dropout=0.)
    directory = tmp_path / 'base'
    LlamaForCausalLM(config).save_pretrained(directory, safe_serialization=True)
    data = tmp_path / 'data.jsonl'
    data.write_text('\n'.join(json.dumps({'input_ids': [1, 2, 3 + i, 8], 'labels': [-100, -100, 3 + i, 8]})
                              for i in range(4)) + '\n', encoding='utf-8')
    args = argparse.Namespace(model=directory, data=data, output=tmp_path / 'resume-run',
                              base_id='test-local-base', backbone='model', head='lm_head',
                              targets=['model.layers.0.self_attn.q_proj'], checkpoint_modules=None,
                              auto_class='AutoModelForCausalLM', attention='eager', cpu_reference=True,
                              compile=compiled,
                              no_preserve_rng_state=False, chat=False, train_on_prompt=False,
                              truncate=False, pad_token_id=0, max_length=8, steps=1, batch_size=1,
                              accumulation=1, checkpoint_every=1, token_chunk_size=2,
                              rank=2, alpha=2., lr=.005, seed=115, dtype='fp32', resume=None)
    example.run(args)
    args.resume = args.output / 'checkpoints/latest.pt'; args.steps = 2
    continued = example.run(args)
    saved = torch.load(args.resume, weights_only=True)
    args.resume = None; args.output = tmp_path / 'uninterrupted'
    uninterrupted = example.run(args)
    assert continued['loss'] == uninterrupted['loss']
    assert continued['microbatch_cursor'] == 2 and saved['data_state']['cursor'] == 2
    final = torch.load(args.output / 'checkpoints/latest.pt', weights_only=True)
    for name, entry in saved['adapter']['layers'].items():
        for key in ('lora_A', 'lora_B'):
            torch.testing.assert_close(entry[key], final['adapter']['layers'][name][key], rtol=0, atol=0)
