"""Content-addressed artifact store for Cat's Eye.

Layout under the user's data folder:
    artifacts/<stage>/<fingerprint[:2]>/<fingerprint>/
Containing:
    output files
    manifest.json:
        {
            "fingerprint": str,
            "stage": str,
            "version": int,
            "created_at": float,
            "size_bytes": int,
            "sha256_map": dict[str, str],
            "metadata": dict
        }

Atomic write: writes to .tmp_<fingerprint> next to the destination, flushes/fsyncs,
and atomically renames (os.replace) into place. If destination already exists, reuses it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import sqlalchemy as sa
from studio import config, ctx, db

logger = logging.getLogger("studio.artifacts")

_SEG_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _validate_segment(value: str, label: str) -> str:
    s = str(value or "").strip()
    if not _SEG_RE.fullmatch(s) or ".." in s or s in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r}")
    return s


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    """Manages content-addressed artifacts for a specific user directory."""

    def __init__(self, user_dir: Optional[Union[str, Path]] = None, user_id: str = ""):
        if user_dir is not None:
            self.user_dir = Path(user_dir)
        else:
            uid = user_id or ctx.uid_or_none() or "default"
            try:
                self.user_dir = config.user_dir(uid)
            except Exception:
                self.user_dir = config.DATA_DIR / "users" / uid
        self.user_id = user_id or (self.user_dir.name if self.user_dir.parent.name == "users" else "default")
        self.root = self.user_dir / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _dest_dir(self, stage: str, fingerprint: str) -> Path:
        stg = _validate_segment(stage, "stage")
        fp = _validate_segment(fingerprint, "fingerprint")
        if len(fp) < 2:
            raise ValueError(f"Invalid fingerprint length: {fingerprint!r}")
        dest = (self.root / stg / fp[:2] / fp).resolve()
        if not dest.is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path escapes store root")
        return dest

    def _find_dest_dir(self, fingerprint: str, stage: Optional[str] = None) -> Optional[Path]:
        fp = _validate_segment(fingerprint, "fingerprint")
        if len(fp) < 2:
            raise ValueError(f"Invalid fingerprint length: {fingerprint!r}")
        root_res = self.root.resolve()
        if stage:
            dest = self._dest_dir(stage, fp)
            if dest.is_dir() and (dest / "manifest.json").is_file():
                return dest
            return None
        # If stage not specified, check DB first scoped strictly to self.user_id
        try:
            with db.engine.connect() as conn:
                r = conn.execute(
                    sa.select(db.artifacts.c.path, db.artifacts.c.stage).where(
                        sa.and_(db.artifacts.c.fingerprint == fp, db.artifacts.c.user_id == self.user_id)
                    )
                ).first()
                if r:
                    p = Path(r[0]).resolve()
                    if p.is_relative_to(root_res) and p.is_dir() and (p / "manifest.json").is_file():
                        return p
        except Exception:
            pass
        # Search stage folders under this user's artifact root
        if self.root.is_dir():
            for stage_dir in self.root.iterdir():
                if stage_dir.is_dir():
                    cand = (stage_dir / fp[:2] / fp).resolve()
                    if cand.is_relative_to(root_res) and cand.is_dir() and (cand / "manifest.json").is_file():
                        return cand
        return None

    def get(self, fingerprint: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve an artifact manifest if it exists, updating its access time."""
        dest = self._find_dest_dir(fingerprint, stage)
        if not dest:
            return None
        manifest_file = dest / "manifest.json"
        if not manifest_file.is_file():
            return None
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            # Attach local directory path for easy consumer access
            data["_dir"] = str(dest)
            self.touch(fingerprint, stage=stage)
            return data
        except Exception as e:
            logger.warning("Error reading artifact manifest %s: %s", manifest_file, e)
            return None

    def put(
        self,
        stage: str,
        fingerprint: str,
        files: Dict[str, Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        """Atomically store files and manifest under the given stage and fingerprint.

        If the destination already exists, it is reused and the existing manifest is returned.
        """
        stg = _validate_segment(stage, "stage")
        fp = _validate_segment(fingerprint, "fingerprint")
        metadata = metadata or {}
        dest = self._dest_dir(stg, fp)
        parent = dest.parent
        parent.mkdir(parents=True, exist_ok=True)

        if dest.is_dir() and (dest / "manifest.json").is_file():
            # Destination already exists: reuse it
            return self.get(fp, stage=stg) or {}

        # Atomic temp directory
        tmp_dir = (parent / f".tmp_{fp}_{os.getpid()}_{int(time.time() * 1000)}").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            sha256_map = {}
            total_bytes = 0

            for out_name, src in files.items():
                safe_name = _validate_segment(out_name, "artifact output name")
                src_path = Path(src)
                target_path = (tmp_dir / safe_name).resolve()
                if not target_path.is_relative_to(tmp_dir):
                    raise ValueError(f"Output path escapes artifact dir: {out_name!r}")
                if src_path.resolve() != target_path:
                    shutil.copy2(str(src_path), str(target_path))
                # Flush and fsync
                with target_path.open("rb") as f:
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                file_hash = _sha256_file(target_path)
                sha256_map[safe_name] = file_hash
                total_bytes += target_path.stat().st_size

            now = time.time()
            manifest = {
                "fingerprint": fingerprint,
                "stage": stage,
                "version": version,
                "created_at": now,
                "size_bytes": total_bytes,
                "sha256_map": sha256_map,
                "metadata": metadata,
            }
            manifest_file = tmp_dir / "manifest.json"
            manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            with manifest_file.open("rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            # Check if another process created dest in the meantime
            if dest.is_dir() and (dest / "manifest.json").is_file():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                return self.get(fingerprint, stage=stage) or manifest

            # Atomic rename into place
            os.replace(str(tmp_dir), str(dest))

            # Record in DB artifacts table scoped to self.user_id
            try:
                with db.engine.begin() as conn:
                    conn.execute(
                        db.artifacts.delete().where(
                            sa.and_(db.artifacts.c.fingerprint == fp, db.artifacts.c.user_id == self.user_id)
                        )
                    )
                    conn.execute(
                        db.artifacts.insert().values(
                            fingerprint=fp,
                            stage=stg,
                            version=version,
                            user_id=self.user_id,
                            path=str(dest),
                            size_bytes=total_bytes,
                            created_at=now,
                            accessed_at=now,
                            metadata_json=json.dumps(metadata),
                        )
                    )
            except Exception as e:
                logger.warning("Could not record artifact in DB: %s", e)

            manifest["_dir"] = str(dest)
            return manifest
        except Exception:
            # Clean up temp on failure
            if tmp_dir.is_dir():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise

    def touch(self, fingerprint: str, stage: Optional[str] = None) -> bool:
        """Update the accessed_at time of an artifact."""
        dest = self._find_dest_dir(fingerprint, stage)
        if not dest:
            return False
        now = time.time()
        try:
            os.utime(str(dest), (now, now))
            manifest_file = dest / "manifest.json"
            if manifest_file.is_file():
                os.utime(str(manifest_file), (now, now))
        except OSError:
            pass
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    db.artifacts.update()
                    .where(sa.and_(db.artifacts.c.fingerprint == fingerprint, db.artifacts.c.user_id == self.user_id))
                    .values(accessed_at=now)
                )
        except Exception:
            pass
        return True

    def delete(self, fingerprint: str, stage: Optional[str] = None) -> bool:
        """Delete an artifact from disk and remove its DB record."""
        dest = self._find_dest_dir(fingerprint, stage)
        if dest and dest.is_dir():
            shutil.rmtree(str(dest), ignore_errors=True)
        try:
            with db.engine.begin() as conn:
                conn.execute(
                    db.artifacts.delete().where(
                        sa.and_(db.artifacts.c.fingerprint == fingerprint, db.artifacts.c.user_id == self.user_id)
                    )
                )
        except Exception:
            pass
        return True

    def user_disk_usage(self, user_id: Optional[str] = None) -> int:
        """Calculate total disk space used by artifacts for a user."""
        uid = user_id or self.user_id
        # Try DB first
        try:
            with db.engine.connect() as conn:
                total = conn.execute(
                    sa.select(sa.func.sum(db.artifacts.c.size_bytes)).where(
                        db.artifacts.c.user_id == uid
                    )
                ).scalar()
                if total is not None:
                    return int(total)
        except Exception:
            pass

        # Fallback to scanning disk
        total = 0
        if self.root.is_dir():
            for root_dir, _, files in os.walk(str(self.root)):
                for f in files:
                    fp = os.path.join(root_dir, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        return total


# Module-level convenience functions
def get(fingerprint: str, stage: Optional[str] = None, user_dir: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    return ArtifactStore(user_dir=user_dir).get(fingerprint, stage=stage)


def put(
    stage: str,
    fingerprint: str,
    files: Dict[str, Union[str, Path]],
    metadata: Optional[Dict[str, Any]] = None,
    version: int = 1,
    user_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    return ArtifactStore(user_dir=user_dir).put(stage, fingerprint, files, metadata, version=version)


def touch(fingerprint: str, stage: Optional[str] = None, user_dir: Optional[Union[str, Path]] = None) -> bool:
    return ArtifactStore(user_dir=user_dir).touch(fingerprint, stage=stage)


def delete(fingerprint: str, stage: Optional[str] = None, user_dir: Optional[Union[str, Path]] = None) -> bool:
    return ArtifactStore(user_dir=user_dir).delete(fingerprint, stage=stage)


def user_disk_usage(user_id_or_dir: Union[str, Path]) -> int:
    if isinstance(user_id_or_dir, Path) or ("/" in str(user_id_or_dir) or "\\" in str(user_id_or_dir)):
        return ArtifactStore(user_dir=user_id_or_dir).user_disk_usage()
    return ArtifactStore(user_id=str(user_id_or_dir)).user_disk_usage()
