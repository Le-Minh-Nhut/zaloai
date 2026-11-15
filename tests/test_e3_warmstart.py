import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from lifemate.training.lora_finetuning import configure_lora


def cfg():
    return SimpleNamespace(lora=SimpleNamespace(r=2,alpha=4,dropout=0.05,bias='none',target_modules=['q_proj','v_proj']), training=SimpleNamespace(gradient_checkpointing=True),model=SimpleNamespace(name='mini-test'))


def test_real_peft_warmstart_preserves_weights_and_freezes_base(tmp_path):
    c=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
    parent=configure_lora(LlamaForCausalLM(c),cfg())['model']
    with torch.no_grad():
        for n,p in parent.named_parameters():
            if 'lora_' in n:p.fill_(0.123)
    parent.save_pretrained(tmp_path)
    warm=configure_lora(LlamaForCausalLM(c),cfg(),init_adapter=str(tmp_path))['model']
    actual=[p for n,p in warm.named_parameters() if p.requires_grad]
    assert actual and all(torch.allclose(p,torch.full_like(p,0.123)) for p in actual)
    assert all(not p.requires_grad for n,p in warm.named_parameters() if 'lora_' not in n)
    incompatible=cfg();incompatible.lora.r=4
    with pytest.raises(Exception,match='configuration differs'):configure_lora(LlamaForCausalLM(c),incompatible,init_adapter=str(tmp_path))


def test_exposure_balancing_deterministic_capped_and_unchanged():
    path=Path(__file__).resolve().parents[1]/'scripts/train.py';spec=importlib.util.spec_from_file_location('train_e3_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    clean=[{'target':str(i),'meta':{}} for i in range(100)]
    conditioned=[{'target':str(i),'meta':{'augmentation':'conditioned_multi_asr'}} for i in range(10)]
    records=clean+conditioned
    a,info=module.conditioned_exposures(records,.2,42);b,_=module.conditioned_exposures(records,.2,42)
    assert a==b and a[:len(records)]==records and info['unique_conditioned']==10
    assert len(a)==125 and info['planned_exposure_ratio']==.2
    assert all(sum(x is rec for x in a)<=5 for rec in conditioned)
