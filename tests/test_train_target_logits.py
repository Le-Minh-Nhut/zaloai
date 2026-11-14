import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def test_target_only_logits_match_full_loss_and_gradients():
    from transformers import LlamaConfig, LlamaForCausalLM
    from lifemate.training.trainer import _forward_loss, chunked_causal_lm_loss
    torch.manual_seed(41)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=64))
    model.eval()
    ids = torch.randint(0, 48, (2, 18))
    labels = ids.clone()
    labels[0, :13] = -100
    labels[1, :11] = -100
    full = chunked_causal_lm_loss(model(input_ids=ids).logits, labels, 4)
    full.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    sliced = _forward_loss(model, {'input_ids': ids, 'labels': labels}, 4)
    sliced.backward()
    assert torch.allclose(full, sliced, atol=1e-6, rtol=1e-6)
    for name, param in model.named_parameters():
        if name in grads:
            assert torch.allclose(grads[name], param.grad, atol=1e-6, rtol=1e-5), name


def test_nonfinite_gradients_stop_before_optimizer_step():
    from lifemate.training.trainer import SFTTrainer
    trainer = object.__new__(SFTTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.grad_clip = 1.0
    trainer.scaler = None
    trainer.model.weight.grad = torch.full_like(trainer.model.weight, float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        trainer._clip_gradients()


def test_evaluation_suffix_matches_full_logits_with_padding():
    from transformers import LlamaConfig, LlamaForCausalLM
    from lifemate.training.collator import SupervisedCollator
    from lifemate.training.trainer import SFTTrainer
    torch.manual_seed(23)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=2, max_position_embeddings=64))
    class FullLogits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask)
    examples = [
        {'input_ids': list(range(1, 19)), 'labels': [-100] * 13 + list(range(14, 19)), 'attention_mask': [1] * 18},
        {'input_ids': list(range(1, 15)), 'labels': [-100] * 9 + list(range(10, 15)), 'attention_mask': [1] * 14},
    ]
    def evaluate(m):
        trainer = object.__new__(SFTTrainer)
        trainer.model, trainer.micro_batch, trainer.device = m, 2, 'cpu'
        trainer.loss_chunk_tokens, trainer.precision = 4, 'float32'
        trainer.collator = SupervisedCollator(pad_token_id=0)
        return trainer.evaluate(examples)
    full = evaluate(FullLogits(model))
    optimized = evaluate(model)
    assert full == optimized
    assert optimized['val_target_tokens'] == 10
    assert optimized['val_loss'] is not None


def test_checkpoint_cadence_100_retains_latest_two(tmp_path):
    from lifemate.training.trainer import CheckpointManager, SFTTrainer
    trainer = object.__new__(SFTTrainer)
    trainer.log_every = 10
    assert not trainer._due('steps', 99, 100)
    assert trainer._due('steps', 100, 100)
    assert not trainer._due('steps', 101, 100)
    def save_stub(path, payload):
        (path / 'adapter.marker').write_text('test')
        return path
    manager = CheckpointManager(tmp_path, run_name='run', save_total_limit=2, save_fn=save_stub)
    for step in (100, 200, 300):
        manager.save(step, {'state': {'step': step}})
    assert sorted(p.name for p in manager.root.iterdir()) == ['step-200', 'step-300']
    assert manager.latest().name == 'step-300'
