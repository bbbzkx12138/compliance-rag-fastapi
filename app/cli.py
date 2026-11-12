from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from app.config import APP_API_BASE_URL, DATABASE_PATH, POLICIES_PATH, load_policies
from app.services.index_manager import IndexManager
from app.services.repository import PolicyRepository


def build_runtime(
    database_path: Path,
    policies_path: Path,
) -> tuple[PolicyRepository, IndexManager, dict[str, Any], dict[str, Any]]:
    policies = load_policies(policies_path)
    repository = PolicyRepository(database_path)
    seed = repository.sync_builtin_documents(policies["documents"])
    index_manager = IndexManager(policies, repository.list_documents())
    return repository, index_manager, policies, seed


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def rebuild_running_server() -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"{APP_API_BASE_URL}/api/index/rebuild",
        method="POST",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PolicyLens database and index maintenance commands"
    )
    parser.add_argument(
        "command",
        choices=["status", "seed", "rebuild-index"],
        help="Maintenance command",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="SQLite database path",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=POLICIES_PATH,
        help="Seed policy JSON path",
    )
    args = parser.parse_args()

    repository, index_manager, _, seed = build_runtime(
        args.database.resolve(),
        args.policies.resolve(),
    )
    payload = {
        "command": args.command,
        "database": str(args.database.resolve()),
        "database_stats": repository.stats(),
        "seed": seed,
        "index": index_manager.metadata(),
    }
    if args.command == "rebuild-index":
        live_result = rebuild_running_server()
        if live_result is not None:
            payload["mode"] = "running_server"
            payload["server_url"] = APP_API_BASE_URL
            payload["index"] = live_result.get("meta", live_result)
            payload["database_stats"] = live_result.get(
                "database", payload["database_stats"]
            )
        else:
            index_manager.rebuild(
                repository.list_documents(),
                reason="cli_offline_validation",
            )
            payload["mode"] = "offline_validation"
            payload["note"] = (
                "服务未运行，已完成离线索引校验；"
                "服务下次启动时会自动重建进程内索引。"
            )
            payload["index"] = index_manager.metadata()
    print_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
