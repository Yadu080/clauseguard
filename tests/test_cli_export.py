import json

from src.__main__ import main
from src.analyzer import analyze
from src.export import result_to_markdown


def test_cli_exit_codes(capsys):
    assert main(["Candidates must submit the form.", "Candidates may submit the form."]) == 1
    assert "MOD-WEAKENED" in capsys.readouterr().out
    assert main(["Pay the fee.", "Pay the fee."]) == 0


def test_cli_json(capsys):
    main(["Pay 75% of the fee.", "Pay 70% of the fee.", "--json"])
    assert json.loads(capsys.readouterr().out)["findings"][0]["rule_id"] == "NUM-CHANGED"


def test_cli_passage_files(tmp_path, capsys):
    (tmp_path / "o.txt").write_text("Submit the form before Friday. Parents will be informed.")
    (tmp_path / "r.txt").write_text("Submit the form before Friday.")
    code = main([str(tmp_path / "o.txt"), str(tmp_path / "r.txt"), "--passage"])
    assert code == 2  # only a REVIEW: the dropped sentence has no rule markers
    assert "PSG-SENTENCE-DROPPED" in capsys.readouterr().out


def test_markdown_report_contains_evidence():
    md = result_to_markdown(analyze("Students cannot enter unless approved.", "Students cannot enter."))
    assert "COND-EXCEPTION-REMOVED" in md and "**unless**" in md and "not proof" in md
