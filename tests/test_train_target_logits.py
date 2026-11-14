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
