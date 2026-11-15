import importlib.util,json
from pathlib import Path
import pytest
from lifemate.errors import LifeMateError
P=Path(__file__).resolve().parents[1]/'scripts/transcribe_full_multi.py'
spec=importlib.util.spec_from_file_location('full_multi_test',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def line(row):
    return json.dumps({**row,'record_sha256':mod.digest(row)})+'\n'

def test_resume_discards_incomplete_tail_and_retains_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'executed'}
    p.write_text(line(row)+'{"incomplete"')
    assert mod.read_journal(p,'r')[('a','current')]==row
    assert p.read_text()==line(row)

def test_resume_rejects_corruption_or_duplicate_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'executed'}
    p.write_text(line(row)+line(row))
    with pytest.raises(LifeMateError,match='duplicates'):mod.read_journal(p,'r')
    p.write_text(line(row).replace('executed','failed'))
    with pytest.raises(LifeMateError,match='checksum'):mod.read_journal(p,'r')

def test_retry_can_replace_failure_but_not_success(tmp_path):
    p=tmp_path/'rows';row={'sample_id':'a','turn_id':'current','run_identity':'r','status':'failed'}
    success={**row,'status':'executed'};p.write_text(line(row)+line(success))
    assert mod.read_journal(p,'r')[('a','current')]['status']=='executed'
