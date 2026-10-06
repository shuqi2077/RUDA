"""Real two-process CPU/Gloo checks of the shared replica/SFT algorithms."""
from datetime import timedelta
import copy
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from architecture_test_utils import NAME

replicas = importlib.import_module(NAME + '.distributed_training')
cft = importlib.import_module(NAME + '.causal_finetuning')
ft = importlib.import_module(NAME + '.finetuning')


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 4)
        self.projection = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask):
        return self.projection(self.embedding(input_ids))


def batch(rank):
    tokens = torch.tensor([[1, 2, 3, 4]] if rank == 0 else [[2, 4, 5, 6]])
    labels = tokens.clone()
    labels[:, :1 if rank == 0 else 2] = -100
    return {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens, dtype=torch.bool), 'labels': labels}


def worker(rank, uri, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(41 + rank)
        model = cft.CausalLMFinetuner(Backbone(), nn.Linear(4, 13), activation_checkpointing=False)
        ft.inject_lora(model, target_modules=['backbone.projection'], rank=2, alpha=2.)
        group = replicas.ReplicaGroup(device_type='cpu')
        group.initialize(model)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, foreach=False)
        golden_optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=.01, foreach=False)
        trainer = cft.SFTTrainer(model, optimizer, base_id='replica-reference', run_config={}, replica_group=group)
        for step in range(2):
            local = [batch(rank)] if step == 0 or rank == 1 else []
            metrics = trainer.train_step(local)
            golden_optimizer.zero_grad(set_to_none=True)
            batches = [batch(0), batch(1)] if step == 0 else [batch(1)]
            count = sum(int((b['labels'][:, 1:] != -100).sum()) for b in batches)
            total = sum(reference(**b, reduction='sum') for b in batches)
            (total / count).backward()
            golden_optimizer.step()
            assert metrics['supervised_tokens'] == count
            assert metrics['loss'] == pytest.approx(float(total.detach()) / count, rel=2e-6)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        path = trainer.save(Path(directory) / str(rank))
        assert torch.load(path, weights_only=True)['replica'] == {'rank': rank, 'world_size': 2}
        trainer.resume(path)
        assert trainer.step == 2
        # No fake communicator: mismatch reaches and fails on both real processes.
        with pytest.raises(ValueError, match='options differ'):
            group.validate_training_options({'step': rank})
    finally:
        dist.destroy_process_group()


def test_two_process_token_weighted_sft_and_rank_checkpoint(tmp_path):
    uri = (tmp_path / 'rendezvous').resolve().as_uri()
    multiprocessing.spawn(worker, args=(uri, str(tmp_path / 'checkpoint')), nprocs=2, join=True)
