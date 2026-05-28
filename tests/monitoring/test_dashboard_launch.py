import json
from pathlib import Path

from scripts.run_dashboard import build_dashboard, main


def test_build_dashboard_is_local_only() -> None:
    dashboard = build_dashboard()

    assert dashboard.host == "127.0.0.1"
    assert dashboard.port == 8501
    assert dashboard.is_local_only is True


def test_run_dashboard_main_prints_context(capsys) -> None:
    exit_code = main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["streamlit_server"]["server.address"] == "127.0.0.1"
    assert payload["streamlit_server"]["server.port"] == 8501


def test_dashboard_batch_launcher_exists() -> None:
    launcher = Path("start_dashboard.bat")

    assert launcher.exists()
    content = launcher.read_text()
    assert "streamlit run scripts/dashboard_app.py" in content
    assert "--server.address 127.0.0.1" in content
