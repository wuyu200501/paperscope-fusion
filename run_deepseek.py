"""Run the live evaluation or local web app with a session-only credential."""

import argparse
import getpass
import os
import sys
from pathlib import Path

from embedding_models import EMBEDDINGS

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
        "--server.fileWatcherType", "none",
    ]
    cli.main()


def check_key():
    from openai import APIConnectionError, APIStatusError, OpenAI

    try:
        with OpenAI(
            api_key=os.environ["PAPERSCOPE_API_KEY"],
            base_url=os.environ["PAPERSCOPE_BASE_URL"],
            timeout=15,
            max_retries=0,
        ) as client:
            client.models.list()
    except APIStatusError as error:
        if error.status_code == 401:
            raise SystemExit("DeepSeek HTTP 401: API key rejected.") from None
        raise SystemExit(f"DeepSeek returned HTTP {error.status_code}.") from None
    except APIConnectionError:
        raise SystemExit("Could not connect to DeepSeek. Check your network and retry.") from None
    print("DeepSeek HTTP 200: API key accepted.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    evaluations = parser.add_mutually_exclusive_group()
    evaluations.add_argument("--evaluate", action="store_true")
    evaluations.add_argument("--claim-ablation", action="store_true",
                             help="Run the four Direct/Checked × chunking configurations.")
    parser.add_argument("--embedding", choices=EMBEDDINGS, default="small",
                        help="BGE variant for the selected evaluation.")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--check-key", action="store_true", help="Prompt for a fresh key and check it with DeepSeek.")
    parser.add_argument("--use-env-key", action="store_true",
                        help="With --check-key, read PAPERSCOPE_API_KEY instead of prompting.")
    parser.add_argument("--offline", action="store_true", help="Serve local extractive QA.")
    parser.add_argument("--port", type=int, default=8517)
    args = parser.parse_args()
    if not (args.evaluate or args.claim_ablation or args.serve or args.check_key):
        parser.error("choose --evaluate, --claim-ablation, --serve or --check-key")
    if args.offline and (args.evaluate or args.claim_ablation or args.check_key):
        parser.error("evaluations and key checks require DeepSeek")
    if args.use_env_key and not args.check_key:
        parser.error("--use-env-key requires --check-key")
    if args.offline:
        os.environ.pop("PAPERSCOPE_API_KEY", None)
    else:
        api_key = os.environ.get("PAPERSCOPE_API_KEY", "").strip()
        if args.use_env_key:
            if not api_key:
                parser.error("PAPERSCOPE_API_KEY is empty.")
        elif args.check_key or not api_key:
            if not sys.stdin.isatty():
                parser.error("start in an interactive terminal to enter the API key without echo")
            api_key = getpass.getpass("DeepSeek API key: ").strip()
        if not api_key:
            parser.error("No DeepSeek API key was entered.")
        if any(not 33 <= ord(char) <= 126 for char in api_key):
            parser.error("The API key contains non-ASCII, whitespace or control characters. Copy the complete key and retry.")
        os.environ["PAPERSCOPE_API_KEY"] = api_key
    if args.check_key:
        check_key()
    if args.evaluate or args.claim_ablation:
        from evaluate import run

        metrics = run(claims_only=args.claim_ablation, embedding=args.embedding)
        if any(row["api_errors"] for row in metrics["metrics"].values()):
            raise SystemExit("Some requests failed. Rerun the same command to retry failed cases.")
    if args.serve:
        print(f"PaperScope on http://127.0.0.1:{args.port}", flush=True)
        serve(args.port)
