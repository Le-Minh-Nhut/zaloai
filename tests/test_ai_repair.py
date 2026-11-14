import json
from pathlib import Path
import subprocess
import sys

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/ai_repair.py'


def test_repair_refuses_unconfirmed_bug_without_creating_worktree(tmp_path):
    review=tmp_path/'review.json'; review.write_text(json.dumps({'status':'REJECTED','review':{'verdict':'MODEL_UNDERPERFORMANCE'}}))
    worktree=tmp_path/'repair'
    result=subprocess.run([sys.executable,str(SCRIPT),'--review',str(review),'--worktree',str(worktree),'--suite','training'],capture_output=True,text=True)
    assert result.returncode != 0 and 'BUG_CONFIRMED' in result.stderr
    assert not worktree.exists()


def test_isolated_repair_runs_focused_tests_without_touching_source(tmp_path):
    import os
    repo=tmp_path/'repo'; (repo/'scripts').mkdir(parents=True); (repo/'tests').mkdir(); (repo/'src').mkdir()
    (repo/'scripts/ai_repair.py').write_text(SCRIPT.read_text())
    (repo/'tests/test_training.py').write_text('def test_baseline():\n    assert True\n')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','config','user.name','test'],cwd=repo,check=True)
    subprocess.run(['git','config','user.email','test@example.invalid'],cwd=repo,check=True)
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','baseline'],cwd=repo,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    report=tmp_path/'bug.json';report.write_text(json.dumps({'status':'REJECTED','review':{'verdict':'BUG_CONFIRMED'},'facts':{'git_sha':sha}}))
    fake=tmp_path/'bin';fake.mkdir()
    codex=fake/'codex';codex.write_text('#!/bin/sh\nmkdir -p src\nprintf "value = 1\\n" > src/fix.py\nprintf "def test_fix():\\n    assert True\\n" > tests/test_fix.py\n');codex.chmod(0o755)
    env=os.environ.copy();env['PATH']=str(fake)+os.pathsep+env['PATH']
    target=tmp_path/'repair'
    result=subprocess.run([sys.executable,str(repo/'scripts/ai_repair.py'),'--review',str(report),'--worktree',str(target),'--suite','training','--python',sys.executable],cwd=repo,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert (target/'src/fix.py').is_file() and (target/'tests/test_fix.py').is_file()
    assert not (repo/'src/fix.py').exists() and not (repo/'tests/test_fix.py').exists()
    assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)==''
