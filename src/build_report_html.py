"""Render the visual project report from the generated numbers.

Reads reports/report_data.json (produced by src/build_report_data.py) and
injects it into src/report_template.html, so the page can never show a
figure the pipeline did not actually produce.

Usage:  python src/build_report_html.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "src" / "report_template.html"
DATA = ROOT / "reports" / "report_data.json"
OUT = ROOT / "reports" / "project_report.html"


def main() -> None:
    data = json.loads(DATA.read_text())
    template = TEMPLATE.read_text()
    if "__DATA__" not in template:
        raise SystemExit("template is missing the __DATA__ placeholder")
    # compact JSON keeps the page small; </script> can never appear inside it
    payload = json.dumps(data, separators=(",", ":"))
    OUT.write_text(template.replace("__DATA__", payload))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
