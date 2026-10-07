import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Build script for the application.")
    parser.add_argument(
        "target",
        nargs="?",
        choices=["win", "windows", "mac", "macos"],
        default=None,
        help="Target platform to build for (win/mac). Defaults to current OS.",
    )
    args = parser.parse_args()

    target = args.target
    if not target:
        if sys.platform.startswith("win"):
            target = "win"
        elif sys.platform.startswith("darwin"):
            target = "mac"
        else:
            print(f"Unsupported current platform: {sys.platform}")
            sys.exit(1)

    if target in ["win", "windows"]:
        print("Preparing Windows build...")
        if not sys.platform.startswith("win"):
            print(
                "Note: PyInstaller builds native .exe binaries best when run directly on Windows."
            )
        cmd = [
            "pyinstaller",
            "--distpath",
            "dist/windows",
            "--workpath",
            "build/windows",
            "specs/windows.spec",
        ]
    elif target in ["mac", "macos"]:
        print("Preparing macOS build...")
        if not sys.platform.startswith("darwin"):
            print(
                "Note: PyInstaller builds native .app bundles best when run directly on macOS."
            )
        cmd = [
            "pyinstaller",
            "--distpath",
            "dist/macos",
            "--workpath",
            "build/macos",
            "specs/macos.spec",
        ]
    else:
        print(f"Invalid target specified: {target}")
        sys.exit(1)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
