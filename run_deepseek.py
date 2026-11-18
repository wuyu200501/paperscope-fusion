"""Run the live evaluation or local web app with a session-only credential."""

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.update({
    "HF_HOME": str(CACHE / "huggingface"),
    "HF_HUB_OFFLINE": "1",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "TEMP": str(CACHE),
    "TMP": str(CACHE),
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "PAPERSCOPE_BASE_URL": "https://api.deepseek.com",
    "PAPERSCOPE_MODEL": "deepseek-flash",
})
sys.path.insert(0, str(ROOT / ".deps"))


def serve(port):
    from streamlit.web import cli

    sys.argv = [
        "streamlit", "run", str(ROOT / "app.py"),
        "--global.developmentMode", "false", "--server.address", "127.0.0.1",
        "--server.port", str(port), "--server.headless", "true",
    ]
    cli.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Serve local extractive QA.")
    parser.add_argument("--port", type=int, default=8517)
    args = parser.parse_args()
    if not (args.evaluate or args.serve):
        parser.error("choose --evaluate, --serve or both")
    if args.offline and args.evaluate:
        parser.error("the evaluation requires DeepSeek")
    if args.offline:
        os.environ.pop("PAPERSCOPE_API_KEY", None)
    elif not os.environ.get("PAPERSCOPE_API_KEY"):
        if not sys.stdin.isatty():
            parser.error("start in an interactive terminal to enter the API key without echo")
        os.environ["PAPERSCOPE_API_KEY"] = getpass.getpass("DeepSeek API key: ").strip()
    if args.evaluate:
        from evaluate import run

        run()
    if args.serve:
        print(f"PaperScope on http://127.0.0.1:{args.port}", flush=True)
        serve(args.port)
