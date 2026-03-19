from __future__ import annotations

import sys


def _main() -> int:
    try:
        from src.discussion_app.main import run
    except ImportError as exc:
        message = str(exc)
        if "QtWidgets" in message or "DLL load failed" in message:
            print(
                "Failed to load PySide6 / Qt runtime.\n"
                "Recommended fix for this project:\n"
                "1. Activate the project environment: conda activate myenv\n"
                "2. Reinstall the known-good GUI dependency set:\n"
                "   python -m pip install --force-reinstall PySide6==6.8.3\n"
                "3. Start the app again with: python app.py\n\n"
                f"Original error: {exc}",
                file=sys.stderr,
            )
            return 1
        raise
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
