from __future__ import annotations

import uvicorn

from app.config import (
    APP_HOST,
    APP_LOG_LEVEL,
    APP_PORT,
    APP_RELOAD,
    WEB_CONCURRENCY,
)


def main() -> None:
    if WEB_CONCURRENCY > 1:
        raise SystemExit(
            "当前索引存储在单个 Python 进程中。"
            "请设置 WEB_CONCURRENCY=1，或先接入共享索引服务。"
        )
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_RELOAD,
        workers=1,
        log_level=APP_LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
