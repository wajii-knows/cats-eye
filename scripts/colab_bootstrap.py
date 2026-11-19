#!/usr/bin/env python3
"""Cat's Eye Google Colab Bootstrap Script.

Automates:
1. Google Drive mount verification and directory initialization.
2. Durable model caching (avoids re-downloading 480MB Parakeet model on every session restart).
3. Session lease acquisition and concurrency protection.
4. Database restoration from latest verified checkpoint.
5. Server launch with single-owner authentication.
6. Displaying the official google.colab.kernel.proxyPort(8765) tunnel link.
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_google_colab() -> bool:
    """Detect if running inside an active Google Colab environment."""
    return Path("/content").exists() and (
        "google.colab" in sys.modules or os.path.exists("/opt/bin/nvidia-smi") or "COLAB_GPU" in os.environ
    )


def ensure_drive_mounted(durable_dir: Path) -> bool:
    """Ensure Google Drive is mounted at /content/drive if in Colab."""
    drive_mount = Path("/content/drive")
    if Path("/content").exists() and not (drive_mount / "MyDrive").exists():
        print("  Mounting Google Drive...", flush=True)
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as e:
            print(f"  Warning: Google Drive could not be auto-mounted: {e}")
            print("  Please run: from google.colab import drive; drive.mount('/content/drive')")
            return False

    durable_dir.mkdir(parents=True, exist_ok=True)
    return True


def check_disk_space(path: Path, min_free_gb: float = 3.0) -> float:
    """Verify minimum free space on durable storage."""
    try:
        free_gb = shutil.disk_usage(path).free / 1e9
        if free_gb < min_free_gb:
            print(
                f"  WARNING: Low space on Drive ({free_gb:.1f} GB free, minimum recommended is {min_free_gb:.1f} GB).",
                flush=True,
            )
        return free_gb
    except OSError:
        return 0.0


def sync_models(work_models_dir: Path, durable_models_dir: Path) -> None:
    """Cache models in Drive so runtime wipes don't trigger 500MB re-downloads."""
    work_models_dir.mkdir(parents=True, exist_ok=True)
    durable_models_dir.mkdir(parents=True, exist_ok=True)

    # 1. If model exists in durable store but not in work dir, copy to fast local disk
    for model_dir in durable_models_dir.iterdir():
        if model_dir.is_dir() and not (work_models_dir / model_dir.name).exists():
            print(f"  Restoring speech model '{model_dir.name}' from Google Drive cache...", flush=True)
            shutil.copytree(str(model_dir), str(work_models_dir / model_dir.name))


def get_owner_password() -> str:
    """Get owner password from env, Colab secrets, or generate a new one."""
    pw = os.environ.get("PEAKCUT_OWNER_PASSWORD", "").strip()
    if pw:
        return pw

    # Try Colab userdata secrets
    try:
        from google.colab import userdata  # type: ignore

        pw = (userdata.get("PEAKCUT_OWNER_PASSWORD") or userdata.get("CATS_EYE_PASSWORD") or "").strip()
        if pw:
            return pw
    except Exception:
        pass

    # Generate secure random password
    pw = secrets.token_hex(8)
    print("\n" + "=" * 60)
    print(f"  [SECURITY] Generated Cat's Eye Owner Password: {pw}")
    print("  Save this password to unlock your Colab session in the browser.")
    print("  Or set PEAKCUT_OWNER_PASSWORD in Colab Secrets (key icon).")
    print("=" * 60 + "\n", flush=True)
    return pw


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat's Eye Google Colab Bootstrap")
    parser.add_argument("--work-dir", default="", help="Fast local scratch directory")
    parser.add_argument("--durable-dir", default="", help="Durable Google Drive directory")
    parser.add_argument("--port", type=int, default=8765, help="Web server port")
    parser.add_argument("--force-lease", action="store_true", help="Take over active session lease")
    parser.add_argument("--tunnel", choices=["proxy", "cloudflared"], default="proxy", help="Web UI access method")
    args = parser.parse_args()

    in_colab = check_google_colab()
    print(f"\n========================================================")
    print(f"  CAT'S EYE: GOOGLE COLAB RUNTIME BOOTSTRAP")
    print(f"  Environment: {'Google Colab' if in_colab else 'Local / Self-hosted'}")
    print(f"========================================================\n", flush=True)

    # Set Colab flags
    os.environ["PEAKCUT_COLAB"] = "1"
    os.environ["PEAKCUT_PORT"] = str(args.port)

    # Determine directories
    if in_colab:
        work_dir = Path(args.work_dir or "/content/cats_eye_work")
        durable_dir = Path(args.durable_dir or "/content/drive/MyDrive/CatsEye")
    else:
        work_dir = Path(args.work_dir or ROOT / "data" / "colab_work")
        durable_dir = Path(args.durable_dir or ROOT / "data" / "colab_durable")

    os.environ["PEAKCUT_WORK_DIR"] = str(work_dir)
    os.environ["PEAKCUT_DURABLE_DIR"] = str(durable_dir)

    # Mount Drive
    if in_colab:
        ensure_drive_mounted(durable_dir)

    free_gb = check_disk_space(durable_dir)
    print(f"  Google Drive durable store: {durable_dir} ({free_gb:.1f} GB free)")
    print(f"  Fast local work store:      {work_dir}")

    # Synchronize models
    sync_models(work_dir / "models", durable_dir / "models")

    # Password
    pw = get_owner_password()
    os.environ["PEAKCUT_OWNER_PASSWORD"] = pw

    if args.force_lease:
        os.environ["PEAKCUT_FORCE_LEASE"] = "1"

    # Import studio modules after env vars configured
    from studio import config
    from studio.storage.colab import get_colab_manager

    mgr = get_colab_manager()
    if not mgr:
        print("  Error: Could not initialize ColabManager.")
        sys.exit(1)

    try:
        print("  Acquiring session lease and verifying durable checkpoints...", flush=True)
        mgr.start(force_lease=args.force_lease)
    except Exception as e:
        print(f"\n  [ERROR] Failed to start durable manager: {e}")
        if "Active lease held" in str(e):
            print("  Another Colab notebook session is actively running with this Drive folder.")
            print("  Close the other session or rerun with --force-lease to take it over.\n")
        sys.exit(1)

    status = mgr.get_status()
    rec = status["recovery"]
    if rec["recovered"]:
        print(f"  >>> CHECKPOINT RESTORED: Generation {rec['checkpoint'].get('generation')} verified.")
        print(f"  >>> Auto-resuming {rec['resumed_jobs']} unfinished video job(s).")
    else:
        print("  >>> Clean database initialized. Ready for new jobs.")

    # Expose web access instructions
    print(f"\n  Starting Cat's Eye web server on http://127.0.0.1:{args.port} ...", flush=True)

    if in_colab:
        try:
            from google.colab.output import eval_js  # type: ignore

            proxy_url = eval_js(f"google.colab.kernel.proxyPort({args.port})")
            print("\n" + "*" * 60)
            print("  OPEN CAT'S EYE IN YOUR BROWSER:")
            print(f"  {proxy_url}")
            print(f"  Password: {pw}")
            print("*" * 60 + "\n", flush=True)
        except Exception:
            print(f"  Access via Colab proxy port: http://localhost:{args.port}")

    # Launch server
    cmd = [sys.executable, "-m", "studio.server"]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT))
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\n  Stopping Cat's Eye Colab runtime...", flush=True)
        mgr.stop()
        print("  All data safely committed to Drive. Safe to disconnect.")


if __name__ == "__main__":
    main()
