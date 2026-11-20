"""Durable store abstraction for Cat's Eye in Google Colab and self-hosted environments.

Every durable write follows:
    write to <name>.tmp.<session> -> flush and fsync -> read back and verify size + sha256 -> rename to final -> only then update manifest.
A file without a manifest entry is treated as absent.
Small directories (snapshots, frames, JSON) are packed into a single archive before upload
to prevent Google Drive small-file rate limits and I/O bottlenecks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("studio.storage.durable")


def compute_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class DurableStoreError(Exception):
    pass


class IntegrityError(DurableStoreError):
    pass


class DurableStore:
    """Abstract interface for durable storage."""

    def put_file(self, local_path: Union[str, Path], durable_rel_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def put_dir_packed(
        self,
        local_dir: Union[str, Path],
        durable_rel_path: str,
        archive_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_file(self, durable_rel_path: str, local_dest_path: Union[str, Path]) -> Path:
        raise NotImplementedError

    def get_dir_unpacked(self, durable_rel_path: str, local_dest_dir: Union[str, Path]) -> Path:
        raise NotImplementedError

    def exists(self, durable_rel_path: str) -> bool:
        raise NotImplementedError

    def list(self, prefix: str = "") -> List[str]:
        raise NotImplementedError

    def stat(self, durable_rel_path: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def delete(self, durable_rel_path: str) -> bool:
        raise NotImplementedError

    def commit(self, manifest: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def get_manifest(self) -> Dict[str, Any]:
        raise NotImplementedError


class LocalFolderStore(DurableStore):
    """File-system durable store implementing the verified write protocol and manifest indexing."""

    def __init__(self, root_dir: Union[str, Path], session_id: Optional[str] = None):
        self.root = Path(root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.manifest_path = self.root / "manifest.json"
        self._manifest_lock = threading.RLock()
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.is_file():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read durable manifest: %s", e)
        return {"version": 1, "files": {}, "updated_at": 0.0}

    def get_manifest(self) -> Dict[str, Any]:
        with self._manifest_lock:
            return dict(self._manifest)

    def _save_manifest(self) -> None:
        with self._manifest_lock:
            self._manifest["updated_at"] = time.time()
            tmp = self.root / f"manifest.json.tmp.{self.session_id}_{uuid.uuid4().hex[:8]}"
            tmp.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8")
            with tmp.open("rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            for attempt in range(5):
                try:
                    os.replace(str(tmp), str(self.manifest_path))
                    break
                except OSError:
                    time.sleep(0.05)

    def commit(self, manifest: Optional[Dict[str, Any]] = None) -> None:
        if manifest is not None:
            self._manifest = dict(manifest)
        self._save_manifest()

    def _resolve(self, durable_rel_path: str) -> Path:
        rel = durable_rel_path.replace("\\", "/").lstrip("/")
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ValueError(f"Path escapes durable root: {durable_rel_path}")
        return p

    def put_file(self, local_path: Union[str, Path], durable_rel_path: str) -> Dict[str, Any]:
        src = Path(local_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Local file does not exist: {src}")

        dest = self._resolve(durable_rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        expected_size = src.stat().st_size
        expected_sha = compute_sha256(src)

        tmp_name = f".tmp.{self.session_id}_{uuid.uuid4().hex[:8]}"
        tmp_path = dest.parent / tmp_name

        try:
            # 1. Write to temp file
            shutil.copyfile(str(src), str(tmp_path))

            # 2. Flush and fsync
            with tmp_path.open("rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            # 3. Read back and verify size + sha256
            actual_size = tmp_path.stat().st_size
            if actual_size != expected_size:
                raise IntegrityError(f"Size mismatch on write: expected {expected_size}, got {actual_size}")

            actual_sha = compute_sha256(tmp_path)
            if actual_sha != expected_sha:
                raise IntegrityError(f"SHA-256 mismatch on write: expected {expected_sha}, got {actual_sha}")

            # 4. Atomic rename into final location
            os.replace(str(tmp_path), str(dest))

            # 5. Update manifest and commit
            norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
            meta = {
                "size": expected_size,
                "sha256": expected_sha,
                "created_at": time.time(),
                "session_id": self.session_id,
            }
            self._manifest["files"][norm_rel] = meta
            self._save_manifest()
            return meta
        except Exception:
            if tmp_path.is_file():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def put_dir_packed(
        self,
        local_dir: Union[str, Path],
        durable_rel_path: str,
        archive_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        src_dir = Path(local_dir).resolve()
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Local directory does not exist: {src_dir}")

        if durable_rel_path.endswith((".tar", ".tar.gz", ".tgz")):
            rel_target = durable_rel_path
            arch_name = Path(durable_rel_path).name
        else:
            arch_name = archive_name or f"{src_dir.name}.tar"
            rel_target = f"{durable_rel_path.rstrip('/\\')}/{arch_name}"

        with tempfile.TemporaryDirectory(prefix="ce_pack_") as tmp_pack:
            tar_path = Path(tmp_pack) / arch_name
            with tarfile.open(tar_path, "w") as tar:
                tar.add(str(src_dir), arcname=".")

            return self.put_file(tar_path, rel_target)

    def get_file(self, durable_rel_path: str, local_dest_path: Union[str, Path]) -> Path:
        norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
        meta = self._manifest.get("files", {}).get(norm_rel)
        if not meta:
            raise FileNotFoundError(f"Durable artifact not recorded in manifest: {norm_rel}")

        src = self._resolve(norm_rel)
        if not src.is_file():
            raise FileNotFoundError(f"Durable file missing on disk: {src}")

        dest = Path(local_dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_dest = dest.parent / f".tmp.{self.session_id}_{uuid.uuid4().hex[:8]}"
        shutil.copyfile(str(src), str(tmp_dest))

        with tmp_dest.open("rb") as f:
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        # Verify integrity
        if tmp_dest.stat().st_size != meta["size"]:
            tmp_dest.unlink(missing_ok=True)
            raise IntegrityError(f"Corrupted file on read: size mismatch for {norm_rel}")

        act_sha = compute_sha256(tmp_dest)
        if act_sha != meta["sha256"]:
            tmp_dest.unlink(missing_ok=True)
            raise IntegrityError(f"Corrupted file on read: sha256 mismatch for {norm_rel}")

        os.replace(str(tmp_dest), str(dest))
        return dest

    def get_dir_unpacked(self, durable_rel_path: str, local_dest_dir: Union[str, Path]) -> Path:
        norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
        dest_dir = Path(local_dest_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ce_unpack_") as tmp_unpack:
            local_tar = Path(tmp_unpack) / Path(norm_rel).name
            self.get_file(norm_rel, local_tar)
            with tarfile.open(local_tar, "r") as tar:
                safe_members = []
                for m in tar.getmembers():
                    if not (m.isfile() or m.isdir()):
                        continue
                    if os.path.isabs(m.name) or m.name.startswith(("/", "\\")) or ".." in Path(m.name).parts:
                        raise IntegrityError(f"Unsafe tar member path: {m.name!r}")
                    target = (dest_dir / m.name).resolve()
                    if not target.is_relative_to(dest_dir):
                        raise IntegrityError(f"Tar member escapes destination: {m.name!r}")
                    safe_members.append(m)
                try:
                    tar.extractall(path=str(dest_dir), members=safe_members, filter="data")
                except TypeError:
                    tar.extractall(path=str(dest_dir), members=safe_members)
        return dest_dir

    def exists(self, durable_rel_path: str) -> bool:
        norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
        if norm_rel not in self._manifest.get("files", {}):
            return False
        return self._resolve(norm_rel).is_file()

    def list(self, prefix: str = "") -> List[str]:
        norm_prefix = prefix.replace("\\", "/").lstrip("/")
        files = self._manifest.get("files", {})
        return [k for k in sorted(files.keys()) if k.startswith(norm_prefix)]

    def stat(self, durable_rel_path: str) -> Optional[Dict[str, Any]]:
        norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
        return self._manifest.get("files", {}).get(norm_rel)

    def delete(self, durable_rel_path: str) -> bool:
        norm_rel = durable_rel_path.replace("\\", "/").lstrip("/")
        target = self._resolve(norm_rel)
        deleted = False
        if target.is_file():
            try:
                target.unlink(missing_ok=True)
                deleted = True
            except OSError:
                pass
        if norm_rel in self._manifest.get("files", {}):
            del self._manifest["files"][norm_rel]
            self._save_manifest()
            deleted = True
        return deleted


class DriveMountStore(LocalFolderStore):
    """LocalFolderStore with retries and synchronization tailored for Google Drive FUSE mount."""

    def __init__(self, root_dir: Union[str, Path], session_id: Optional[str] = None):
        super().__init__(root_dir, session_id=session_id)

    def put_file(self, local_path: Union[str, Path], durable_rel_path: str) -> Dict[str, Any]:
        last_err = None
        for attempt in range(3):
            try:
                return super().put_file(local_path, durable_rel_path)
            except (OSError, IntegrityError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise DurableStoreError(f"Failed writing to Drive after 3 attempts: {last_err}") from last_err

    def get_file(self, durable_rel_path: str, local_dest_path: Union[str, Path]) -> Path:
        last_err = None
        for attempt in range(3):
            try:
                return super().get_file(durable_rel_path, local_dest_path)
            except (OSError, IntegrityError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise DurableStoreError(f"Failed reading from Drive after 3 attempts: {last_err}") from last_err


class FaultInjectingStore(DurableStore):
    """Test decorator that wraps a DurableStore to inject network latency, truncated writes, or crashes."""

    def __init__(self, wrapped: DurableStore):
        self.wrapped = wrapped
        self.drop_writes = False
        self.truncate_writes_to: Optional[int] = None
        self.crash_before_rename = False
        self.crash_before_manifest_commit = False
        self.inject_latency_sec = 0.0
        self.fail_reads = False
        self.fail_after_n_writes: Optional[int] = None
        self.fail_on_path_substring: Optional[str] = None
        self.on_put_callback: Optional[Any] = None
        self.write_count = 0

    def put_file(self, local_path: Union[str, Path], durable_rel_path: str) -> Dict[str, Any]:
        if self.on_put_callback:
            self.on_put_callback(durable_rel_path)
        if self.fail_on_path_substring and self.fail_on_path_substring in durable_rel_path:
            raise DurableStoreError(f"Simulated write failure on {durable_rel_path}")
        if self.fail_after_n_writes is not None and self.write_count >= self.fail_after_n_writes:
            raise DurableStoreError("Simulated crash after N durable writes")
        if self.inject_latency_sec > 0:
            time.sleep(self.inject_latency_sec)
        if self.drop_writes:
            raise DurableStoreError("Simulated dropped write (network timeout)")

        if self.truncate_writes_to is not None:
            # Emulate truncated write by writing partial bytes
            src = Path(local_path)
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(src.read_bytes()[: self.truncate_writes_to])
                truncated_src = tf.name
            try:
                res = self.wrapped.put_file(truncated_src, durable_rel_path)
                self.write_count += 1
                return res
            finally:
                Path(truncated_src).unlink(missing_ok=True)

        if self.crash_before_rename:
            # Emulate sudden session disconnect while writing to tmp
            if isinstance(self.wrapped, LocalFolderStore):
                dest = self.wrapped._resolve(durable_rel_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dest.parent / f"{dest.name}.tmp.{self.wrapped.session_id}_crash"
                shutil.copyfile(str(local_path), str(tmp_path))
            raise DurableStoreError("Simulated crash before atomic rename")

        if self.crash_before_manifest_commit:
            # Emulate rename succeeded but session disconnected before manifest update
            if isinstance(self.wrapped, LocalFolderStore):
                dest = self.wrapped._resolve(durable_rel_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(local_path), str(dest))
            raise DurableStoreError("Simulated crash before manifest commit")

        res = self.wrapped.put_file(local_path, durable_rel_path)
        self.write_count += 1
        return res

    def put_dir_packed(self, local_dir: Union[str, Path], durable_rel_path: str, archive_name: Optional[str] = None) -> Dict[str, Any]:
        src_dir = Path(local_dir).resolve()
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Local directory does not exist: {src_dir}")
        if durable_rel_path.endswith((".tar", ".tar.gz", ".tgz")):
            rel_target = durable_rel_path
            arch_name = Path(durable_rel_path).name
        else:
            arch_name = archive_name or f"{src_dir.name}.tar"
            rel_target = f"{durable_rel_path.rstrip('/\\')}/{arch_name}"
        with tempfile.TemporaryDirectory(prefix="ce_fault_pack_") as tmp_pack:
            tar_path = Path(tmp_pack) / arch_name
            with tarfile.open(tar_path, "w") as tar:
                tar.add(str(src_dir), arcname=".")
            return self.put_file(tar_path, rel_target)

    def get_file(self, durable_rel_path: str, local_dest_path: Union[str, Path]) -> Path:
        if self.fail_reads:
            raise DurableStoreError("Simulated read failure")
        return self.wrapped.get_file(durable_rel_path, local_dest_path)

    def get_dir_unpacked(self, durable_rel_path: str, local_dest_dir: Union[str, Path]) -> Path:
        return self.wrapped.get_dir_unpacked(durable_rel_path, local_dest_dir)

    def exists(self, durable_rel_path: str) -> bool:
        return self.wrapped.exists(durable_rel_path)

    def list(self, prefix: str = "") -> List[str]:
        return self.wrapped.list(prefix=prefix)

    def stat(self, durable_rel_path: str) -> Optional[Dict[str, Any]]:
        return self.wrapped.stat(durable_rel_path)

    def delete(self, durable_rel_path: str) -> bool:
        return self.wrapped.delete(durable_rel_path)

    def commit(self, manifest: Optional[Dict[str, Any]] = None) -> None:
        self.wrapped.commit(manifest=manifest)

    def get_manifest(self) -> Dict[str, Any]:
        return self.wrapped.get_manifest()
