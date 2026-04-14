"""QIPC transport, shared-memory ring, and binary payload codecs."""

from __future__ import annotations

import atexit
import ctypes
import json
import logging
import os
import select
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


log = logging.getLogger(__name__)


QIPC_MAGIC = b"QIPC"
QIPC_HEADER = struct.Struct("<4sBI")
QIPC_EVAL_REQ = 1
QIPC_EVAL_RESP = 2
QIPC_BATCH_EVAL_REQ = 3
QIPC_BATCH_EVAL_RESP = 4
QIPC_EVAL_REQ_SHM = 5
QIPC_EVAL_RESP_SHM = 6
QIPC_BATCH_EVAL_REQ_SHM = 7
QIPC_BATCH_EVAL_RESP_SHM = 8
QIPC_SHM_LEN = struct.Struct("<I")
QIPC_SHM_DEFAULT_BYTES = 8 * 1024 * 1024

_QIPC_TRANSPORTS = {}
_QIPC_TRANSPORTS_LOCK = threading.Lock()
_SHM_RING_BUFFERS = {}
_SHM_RING_BUFFERS_LOCK = threading.Lock()


def register_qipc_transport(transport):
    key = (transport.req.name, transport.resp.name)
    with _QIPC_TRANSPORTS_LOCK:
        _QIPC_TRANSPORTS[key] = transport


def unregister_qipc_transport(transport):
    key = (transport.req.name, transport.resp.name)
    with _QIPC_TRANSPORTS_LOCK:
        _QIPC_TRANSPORTS.pop(key, None)


def register_ring_buffer(ring):
    with _SHM_RING_BUFFERS_LOCK:
        _SHM_RING_BUFFERS[ring.name] = ring


def unregister_ring_buffer(ring):
    with _SHM_RING_BUFFERS_LOCK:
        _SHM_RING_BUFFERS.pop(ring.name, None)


def cleanup_all_shm():
    with _QIPC_TRANSPORTS_LOCK:
        transports = list(_QIPC_TRANSPORTS.values())
        _QIPC_TRANSPORTS.clear()
    for transport in transports:
        try:
            transport.destroy()
        except Exception:
            pass
    with _SHM_RING_BUFFERS_LOCK:
        rings = list(_SHM_RING_BUFFERS.values())
        _SHM_RING_BUFFERS.clear()
    for ring in rings:
        try:
            ring.destroy()
        except Exception:
            pass


atexit.register(cleanup_all_shm)


def wait_readable(stream, timeout_s):
    """Wait until a stream is readable without select() FD_SETSIZE limits."""
    timeout_ms = max(0, int(float(timeout_s) * 1000.0))
    try:
        poller = select.poll()
        poller.register(stream, select.POLLIN)
        events = poller.poll(timeout_ms)
        return bool(events)
    except (AttributeError, OSError, ValueError):
        ready, _, _ = select.select([stream], [], [], timeout_s)
        return bool(ready)


def _json_line_bytes(payload, json_dumps_compact_fn=None):
    if isinstance(payload, (bytes, bytearray)):
        out = bytes(payload)
    else:
        if json_dumps_compact_fn is None:
            out = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        else:
            out = json_dumps_compact_fn(payload).encode("utf-8")
    return out if out.endswith(b"\n") else out + b"\n"


def _read_exact(stream, n_bytes, timeout_s=None, wait_readable_fn=None):
    chunks = bytearray()
    deadline = None if timeout_s is None else time.perf_counter() + float(timeout_s)
    wait_readable_fn = wait_readable_fn or wait_readable
    while len(chunks) < n_bytes:
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0 or not wait_readable_fn(stream, remaining):
                raise TimeoutError(f"timed out reading {n_bytes} bytes from IPC stream")
        chunk = stream.read(n_bytes - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def proc_write_json_line(proc_or_stream, payload, json_dumps_compact_fn=None):
    stream = getattr(proc_or_stream, "stdin", proc_or_stream)
    stream.write(_json_line_bytes(payload, json_dumps_compact_fn=json_dumps_compact_fn))
    stream.flush()


def proc_write_qipc_frame(proc_or_stream, frame_kind, payload):
    stream = getattr(proc_or_stream, "stdin", proc_or_stream)
    payload = bytes(payload)
    stream.write(QIPC_HEADER.pack(QIPC_MAGIC, int(frame_kind), len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()


def proc_read_json_line(proc_or_stream):
    stream = getattr(proc_or_stream, "stdout", proc_or_stream)
    line = stream.readline()
    if not line:
        return None
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    line = line.strip()
    return line or None


def proc_read_message(proc_or_stream, timeout_s=None, json_loads_fast_fn=None, logger=None):
    logger = logger or log
    stream = getattr(proc_or_stream, "stdout", proc_or_stream)
    first = _read_exact(stream, 1, timeout_s=timeout_s)
    while first in (b"\n", b"\r"):
        first = _read_exact(stream, 1, timeout_s=timeout_s)
    if not first:
        return None, None
    if first in (b"{", b"["):
        rest = stream.readline()
        if isinstance(rest, bytes):
            raw = first + rest
            text = raw.decode("utf-8")
        else:
            text = first.decode("utf-8") + rest
        text = text.strip()
        if not text:
            return "json", None
        try:
            loads = json_loads_fast_fn or json.loads
            return "json", loads(text)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("proc_read_message: JSON parse failed (%s), skipping line", exc)
            return "json", None
    header_rest = _read_exact(stream, QIPC_HEADER.size - 1, timeout_s=timeout_s)
    if header_rest is None:
        return None, None
    try:
        magic, frame_kind, payload_len = QIPC_HEADER.unpack(first + header_rest)
    except struct.error as exc:
        logger.warning("proc_read_message: QIPC header unpack failed (%s), skipping", exc)
        return None, None
    if magic != QIPC_MAGIC:
        logger.warning("proc_read_message: unexpected IPC frame magic: %r", magic)
        return None, None
    if payload_len > 256 * 1024 * 1024:
        logger.warning("proc_read_message: unreasonable payload_len=%d, skipping", payload_len)
        return None, None
    payload = _read_exact(stream, payload_len, timeout_s=timeout_s)
    if payload is None:
        return None, None
    return "frame", (frame_kind, payload)


def proc_decode_eval_frame(proc, frame_kind, payload):
    transport = getattr(proc, "_quartz_qipc_transport", None)
    if frame_kind == QIPC_EVAL_REQ_SHM:
        if transport is None:
            raise RuntimeError("shared-memory eval request received without transport")
        (n_bytes,) = QIPC_SHM_LEN.unpack(payload)
        return QIPC_EVAL_REQ, transport.read_request(n_bytes)
    if frame_kind == QIPC_BATCH_EVAL_REQ_SHM:
        if transport is None:
            raise RuntimeError("shared-memory batch eval request received without transport")
        (n_bytes,) = QIPC_SHM_LEN.unpack(payload)
        return QIPC_BATCH_EVAL_REQ, transport.read_request(n_bytes)
    return frame_kind, payload


def proc_write_eval_response(proc, logical_kind, payload, prefer_shm=False):
    transport = getattr(proc, "_quartz_qipc_transport", None)
    if prefer_shm and transport is not None and transport.write_response(payload):
        shm_kind = {
            QIPC_EVAL_RESP: QIPC_EVAL_RESP_SHM,
            QIPC_BATCH_EVAL_RESP: QIPC_BATCH_EVAL_RESP_SHM,
        }.get(logical_kind)
        if shm_kind is not None:
            proc_write_qipc_frame(proc, shm_kind, QIPC_SHM_LEN.pack(len(payload)))
            return
    proc_write_qipc_frame(proc, logical_kind, payload)


def get_qipc_transport(proc):
    return getattr(proc, "_quartz_qipc_transport", None)


def cleanup_qipc_transport(proc, unregister_ring_buffer_fn=None):
    transport = get_qipc_transport(proc)
    if transport is not None:
        try:
            transport.destroy()
        finally:
            try:
                delattr(proc, "_quartz_qipc_transport")
            except Exception:
                pass
    ring = getattr(proc, "_quartz_ring_buffer", None)
    if ring is not None:
        try:
            ring.destroy()
            if unregister_ring_buffer_fn is None:
                unregister_ring_buffer(ring)
            else:
                unregister_ring_buffer_fn(ring)
        finally:
            try:
                delattr(proc, "_quartz_ring_buffer")
            except Exception:
                pass


def stall_trace_path():
    path = os.environ.get("QUARTZ_STALL_TRACE_PATH", "").strip()
    return path or None


def stall_trace(event, path_fn=None, **fields):
    path = stall_trace_path() if path_fn is None else path_fn()
    if not path:
        return
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "event": str(event),
    }
    for key, value in fields.items():
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        record[str(key)] = value
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass


def launch_rust_server(
    rust_binary,
    *,
    qipc_transport_cls=None,
    shm_ring_buffer_cls=None,
    register_ring_buffer_fn=register_ring_buffer,
    stall_trace_fn=stall_trace,
    popen_fn=subprocess.Popen,
):
    """Start Rust server and fail fast with stderr if it exits immediately."""
    transport = None
    ring_buffer = None
    env = os.environ.copy()
    disable_shm = str(env.get("QUARTZ_DISABLE_QIPC_SHM", "")).strip().lower() in {"1", "true", "yes", "on"}
    qipc_transport_cls = qipc_transport_cls or QipcSharedMemoryTransport
    shm_ring_buffer_cls = shm_ring_buffer_cls or ShmRingBuffer
    try:
        if not disable_shm:
            transport = qipc_transport_cls.create()
            env["QUARTZ_QIPC_REQ_SHM_NAME"] = transport.req.name
            env["QUARTZ_QIPC_RESP_SHM_NAME"] = transport.resp.name
            env["QUARTZ_QIPC_REQ_SHM_SIZE"] = str(transport.size)
            env["QUARTZ_QIPC_RESP_SHM_SIZE"] = str(transport.size)
            try:
                ring_buffer = shm_ring_buffer_cls.create(r2p_slots=2, p2r_slots=2)
                env["QUARTZ_QIPC_RING_SHM_NAME"] = ring_buffer.name
                env["QUARTZ_QIPC_RING_SHM_SIZE"] = str(ring_buffer.size)
                register_ring_buffer_fn(ring_buffer)
            except Exception:
                ring_buffer = None
    except Exception:
        transport = None
    stall_trace_fn("rust_server_launch", rust_binary=rust_binary, shm=bool(transport))
    proc = popen_fn(
        [rust_binary, "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        env=env,
    )
    deadline = time.time() + 0.015
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.005)
    if proc.poll() is not None:
        if ring_buffer is not None:
            ring_buffer.destroy()
        if transport is not None:
            transport.destroy()
        stderr = ""
        try:
            stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        raise RuntimeError(
            f"Rust server exited immediately: {rust_binary}"
            + (f" | stderr: {stderr[:400]}" if stderr else "")
        )
    if transport is not None:
        proc._quartz_qipc_transport = transport
    if ring_buffer is not None:
        proc._quartz_ring_buffer = ring_buffer
    stall_trace_fn("rust_server_ready", child_pid=proc.pid, shm=bool(transport), ring=bool(ring_buffer))
    return proc


def stop_rust_server(
    proc,
    timeout=3.0,
    *,
    write_json_line_fn=proc_write_json_line,
    cleanup_qipc_transport_fn=cleanup_qipc_transport,
    stall_trace_fn=stall_trace,
):
    if proc is None:
        return
    stall_trace_fn("rust_server_stop_begin", child_pid=getattr(proc, "pid", None))
    try:
        write_json_line_fn(proc, {"cmd": "quit"})
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        cleanup_qipc_transport_fn(proc)
        stall_trace_fn(
            "rust_server_stop_end",
            child_pid=getattr(proc, "pid", None),
            returncode=getattr(proc, "returncode", None),
        )


@dataclass
class QipcSharedMemoryTransport:
    req: shared_memory.SharedMemory
    resp: shared_memory.SharedMemory
    size: int

    @classmethod
    def create(cls, size=None):
        size = int(size or os.environ.get("QUARTZ_QIPC_SHM_BYTES", QIPC_SHM_DEFAULT_BYTES))
        transport = cls(
            req=shared_memory.SharedMemory(create=True, size=size),
            resp=shared_memory.SharedMemory(create=True, size=size),
            size=size,
        )
        transport._destroyed = False
        register_qipc_transport(transport)
        return transport

    def close(self):
        for shm in (self.req, self.resp):
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self):
        for shm in (self.req, self.resp):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def destroy(self):
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        self.close()
        self.unlink()
        unregister_qipc_transport(self)

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def read_request(self, n_bytes):
        n_bytes = int(n_bytes)
        if n_bytes < 0 or n_bytes > self.size:
            raise ValueError(f"invalid shared request size: {n_bytes}")
        return bytes(self.req.buf[:n_bytes])

    def write_response(self, payload):
        payload = bytes(payload)
        if len(payload) > self.size:
            return False
        self.resp.buf[:len(payload)] = payload
        return True


SHM_RING_MAGIC = 0x51524E47
SHM_RING_VERSION = 1
SHM_RING_HEADER_SIZE = 256
SHM_RING_SLOT_HEADER = 16
SHM_RING_DEFAULT_SIZE = 16 * 1024 * 1024

SHM_SLOT_EMPTY = 0
SHM_SLOT_WRITTEN = 1
SHM_SLOT_DONE = 2

SHM_MSG_EVAL_BATCH_REQ = 1
SHM_MSG_EVAL_BATCH_RESP = 2
SHM_MSG_JSON = 3
SHM_MSG_SEARCH_RESP = 4

SHM_DIR_TO_PYTHON = 0
SHM_DIR_TO_RUST = 1


@dataclass
class ShmRingBuffer:
    _shm: shared_memory.SharedMemory
    r2p_slot_count: int
    p2r_slot_count: int
    slot_data_size: int
    r2p_base: int
    p2r_base: int

    @classmethod
    def create(cls, r2p_slots=2, p2r_slots=2, slot_data_size=None):
        total_slots = r2p_slots + p2r_slots
        if slot_data_size is None:
            size = int(os.environ.get("QUARTZ_QIPC_RING_SHM_SIZE", SHM_RING_DEFAULT_SIZE))
            slot_data_size = (size - SHM_RING_HEADER_SIZE) // total_slots
        else:
            size = SHM_RING_HEADER_SIZE + total_slots * slot_data_size
        shm = shared_memory.SharedMemory(create=True, size=size)
        struct.pack_into(
            "<IIIII",
            shm.buf,
            0,
            SHM_RING_MAGIC,
            SHM_RING_VERSION,
            r2p_slots,
            p2r_slots,
            slot_data_size,
        )
        struct.pack_into("<IB", shm.buf, 20, 0, 0)
        r2p_base = SHM_RING_HEADER_SIZE
        p2r_base = SHM_RING_HEADER_SIZE + r2p_slots * slot_data_size
        for i in range(total_slots):
            off = SHM_RING_HEADER_SIZE + i * slot_data_size
            shm.buf[off] = SHM_SLOT_EMPTY
        ring = cls(
            _shm=shm,
            r2p_slot_count=r2p_slots,
            p2r_slot_count=p2r_slots,
            slot_data_size=slot_data_size,
            r2p_base=r2p_base,
            p2r_base=p2r_base,
        )
        ring._created = True
        return ring

    @classmethod
    def open(cls, name, size):
        shm = shared_memory.SharedMemory(name=name, create=False, size=size)
        magic, version, r2p_slots, p2r_slots, slot_data_size = struct.unpack_from("<IIIII", shm.buf, 0)
        if magic != SHM_RING_MAGIC or version != SHM_RING_VERSION:
            shm.close()
            return None
        r2p_base = SHM_RING_HEADER_SIZE
        p2r_base = SHM_RING_HEADER_SIZE + r2p_slots * slot_data_size
        ring = cls(
            _shm=shm,
            r2p_slot_count=r2p_slots,
            p2r_slot_count=p2r_slots,
            slot_data_size=slot_data_size,
            r2p_base=r2p_base,
            p2r_base=p2r_base,
        )
        ring._created = False
        return ring

    @property
    def name(self):
        return self._shm.name

    @property
    def size(self):
        return self._shm.size

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass

    def destroy(self):
        try:
            self._shm.close()
        except Exception:
            pass
        if getattr(self, "_created", False):
            try:
                self._shm.unlink()
            except Exception:
                pass

    def _atomic_load_u8(self, offset):
        return ctypes.c_uint8.from_buffer(self._shm.buf, offset).value

    def _atomic_store_u8(self, offset, val):
        ctypes.c_uint8.from_buffer(self._shm.buf, offset).value = val

    def _atomic_load_u32(self, offset):
        return ctypes.c_uint32.from_buffer(self._shm.buf, offset).value

    def _atomic_store_u32(self, offset, val):
        ctypes.c_uint32.from_buffer(self._shm.buf, offset).value = val

    def epoch(self):
        return self._atomic_load_u32(20)

    def cmd_done(self):
        return self._atomic_load_u8(24) != 0

    def request_cancel(self):
        self._atomic_store_u8(25, 1)

    def cancel_requested(self):
        return self._atomic_load_u8(25) != 0

    def _r2p_slot_offset(self, idx):
        return self.r2p_base + idx * self.slot_data_size

    def _p2r_slot_offset(self, idx):
        return self.p2r_base + idx * self.slot_data_size

    def slot_state(self, slot_offset):
        return self._atomic_load_u8(slot_offset)

    def set_slot_state(self, slot_offset, state):
        self._atomic_store_u8(slot_offset, state)

    def r2p_try_read(self, slot_idx):
        off = self._r2p_slot_offset(slot_idx)
        if self.slot_state(off) != SHM_SLOT_WRITTEN:
            return None
        msg_type = self._shm.buf[off + 1]
        payload_len = struct.unpack_from("<I", self._shm.buf, off + 4)[0]
        payload = bytes(self._shm.buf[off + SHM_RING_SLOT_HEADER: off + SHM_RING_SLOT_HEADER + payload_len])
        return msg_type, payload

    def r2p_try_read_meta(self, slot_idx):
        off = self._r2p_slot_offset(slot_idx)
        if self.slot_state(off) != SHM_SLOT_WRITTEN:
            return None
        msg_type = self._shm.buf[off + 1]
        payload_len = struct.unpack_from("<I", self._shm.buf, off + 4)[0]
        epoch = struct.unpack_from("<I", self._shm.buf, off + 8)[0]
        seq = struct.unpack_from("<I", self._shm.buf, off + 12)[0]
        payload = bytes(self._shm.buf[off + SHM_RING_SLOT_HEADER: off + SHM_RING_SLOT_HEADER + payload_len])
        return msg_type, epoch, seq, payload

    def r2p_mark_done(self, slot_idx):
        off = self._r2p_slot_offset(slot_idx)
        self.set_slot_state(off, SHM_SLOT_DONE)

    def p2r_try_write(self, slot_idx, msg_type, payload, epoch=0, seq=0):
        off = self._p2r_slot_offset(slot_idx)
        if len(payload) > self.slot_data_size - SHM_RING_SLOT_HEADER:
            return False
        self._shm.buf[off + 1] = msg_type
        self._shm.buf[off + 2] = SHM_DIR_TO_RUST
        self._shm.buf[off + 3] = 0
        struct.pack_into("<III", self._shm.buf, off + 4, len(payload), epoch, seq)
        self._shm.buf[off + SHM_RING_SLOT_HEADER: off + SHM_RING_SLOT_HEADER + len(payload)] = payload
        self.set_slot_state(off, SHM_SLOT_WRITTEN)
        return True

    def p2r_slot_state(self, slot_idx):
        return self.slot_state(self._p2r_slot_offset(slot_idx))

    def slot_payload_capacity(self):
        return self.slot_data_size - SHM_RING_SLOT_HEADER


QIPC_EVAL_REQ_V2_HEADER = struct.Struct("<IIIQQI")
QIPC_EVAL_REQ_V1_HEADER = struct.Struct("<III")
QIPC_EVAL_REQ_V0_HEADER = struct.Struct("<II")


def unpack_qipc_eval_req(payload):
    if len(payload) < QIPC_EVAL_REQ_V0_HEADER.size:
        raise ValueError("short eval_req payload")
    if len(payload) >= QIPC_EVAL_REQ_V2_HEADER.size:
        model_tag, num_actions, feat_len, fp_lo, fp_hi, encoder_rev = QIPC_EVAL_REQ_V2_HEADER.unpack_from(payload, 0)
        expected_bytes = QIPC_EVAL_REQ_V2_HEADER.size + feat_len * 4
        if len(payload) == expected_bytes:
            features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=QIPC_EVAL_REQ_V2_HEADER.size)
            return num_actions, features, int(model_tag), int(fp_lo), int(fp_hi), int(encoder_rev)
    if len(payload) >= QIPC_EVAL_REQ_V1_HEADER.size:
        model_tag, num_actions, feat_len = QIPC_EVAL_REQ_V1_HEADER.unpack_from(payload, 0)
        expected_bytes = QIPC_EVAL_REQ_V1_HEADER.size + feat_len * 4
        if len(payload) == expected_bytes:
            features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=QIPC_EVAL_REQ_V1_HEADER.size)
            return num_actions, features, int(model_tag), None, None, None
    num_actions, feat_len = QIPC_EVAL_REQ_V0_HEADER.unpack_from(payload, 0)
    expected_bytes = QIPC_EVAL_REQ_V0_HEADER.size + feat_len * 4
    if len(payload) != expected_bytes:
        raise ValueError("eval_req payload length mismatch")
    features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=QIPC_EVAL_REQ_V0_HEADER.size)
    return num_actions, features, 0, None, None, None


def _try_unpack_qipc_batch_eval_req(payload, header):
    (batch_size,) = struct.unpack_from("<I", payload, 0)
    offset = 4
    requests = []
    for _ in range(batch_size):
        if header is QIPC_EVAL_REQ_V2_HEADER:
            if offset + header.size > len(payload):
                return None
            model_tag, num_actions, feat_len, fp_lo, fp_hi, encoder_rev = header.unpack_from(payload, offset)
            offset += header.size
            request_meta = (int(model_tag), int(fp_lo), int(fp_hi), int(encoder_rev))
        elif header is QIPC_EVAL_REQ_V1_HEADER:
            if offset + header.size > len(payload):
                return None
            model_tag, num_actions, feat_len = header.unpack_from(payload, offset)
            offset += header.size
            request_meta = (int(model_tag), None, None, None)
        else:
            if offset + header.size > len(payload):
                return None
            num_actions, feat_len = header.unpack_from(payload, offset)
            offset += header.size
            request_meta = (0, None, None, None)
        byte_len = feat_len * 4
        if offset + byte_len > len(payload):
            return None
        features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=offset)
        offset += byte_len
        model_tag, fp_lo, fp_hi, encoder_rev = request_meta
        requests.append((num_actions, features, model_tag, fp_lo, fp_hi, encoder_rev))
    if offset != len(payload):
        return None
    return requests


def unpack_qipc_batch_eval_req(payload):
    if len(payload) < 4:
        raise ValueError("short batch_eval_req payload")
    for header in (QIPC_EVAL_REQ_V2_HEADER, QIPC_EVAL_REQ_V1_HEADER, QIPC_EVAL_REQ_V0_HEADER):
        requests = _try_unpack_qipc_batch_eval_req(payload, header)
        if requests is not None:
            return requests
    raise ValueError("batch_eval_req payload length mismatch")


def pack_qipc_eval_resp(policy, value):
    policy = np.ascontiguousarray(policy, dtype="<f4")
    return struct.pack("<I", int(policy.size)) + policy.tobytes() + struct.pack("<f", float(value))


def pack_qipc_batch_eval_resp(policies, values):
    payload = bytearray(struct.pack("<I", len(policies)))
    for policy, value in zip(policies, values):
        policy = np.ascontiguousarray(policy, dtype="<f4")
        payload.extend(struct.pack("<I", int(policy.size)))
        payload.extend(policy.tobytes())
        payload.extend(struct.pack("<f", float(value)))
    return bytes(payload)


_SEARCH_RESP_SINGLE = 1
_SEARCH_RESP_MULTI = 2
_SEARCH_RESP_SESSION = 3


def _unpack_search_string(payload, offset):
    if offset + 4 > len(payload):
        raise ValueError("truncated search string length")
    (byte_len,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    if offset + byte_len > len(payload):
        raise ValueError("truncated search string payload")
    raw = payload[offset: offset + byte_len]
    offset += byte_len
    return raw.decode("utf-8"), offset


def unpack_shm_search_response(payload):
    if len(payload) < 13:
        raise ValueError("short search response payload")
    wrapper_kind = payload[0]
    (session_id,) = struct.unpack_from("<Q", payload, 1)
    (result_count,) = struct.unpack_from("<I", payload, 9)
    offset = 13
    results = []
    for _ in range(result_count):
        if offset >= len(payload):
            raise ValueError("truncated search result flags")
        flags = payload[offset]
        offset += 1
        if flags & 0b10:
            results.append(None)
            continue
        if flags & 0b01:
            error, offset = _unpack_search_string(payload, offset)
            results.append({"error": error})
            continue
        if offset + 36 > len(payload):
            raise ValueError("truncated search result scalars")
        best_move, iterations, max_pending = struct.unpack_from("<III", payload, offset)
        offset += 12
        p_flip, value, sigma_q, hbar_eff, dup_rate, avg_vvalue = struct.unpack_from("<ffffff", payload, offset)
        offset += 24
        if offset + 4 > len(payload):
            raise ValueError("truncated sparse policy count")
        (policy_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        policy = []
        for _ in range(policy_len):
            if offset + 8 > len(payload):
                raise ValueError("truncated sparse policy entry")
            idx, prob = struct.unpack_from("<If", payload, offset)
            offset += 8
            policy.append([int(idx), float(prob)])
        if offset + 4 > len(payload):
            raise ValueError("truncated history hash count")
        (history_len,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        history_hashes = []
        for _ in range(history_len):
            if offset + 8 > len(payload):
                raise ValueError("truncated history hash entry")
            (history_hash,) = struct.unpack_from("<Q", payload, offset)
            offset += 8
            history_hashes.append(int(history_hash))
        stop_reason, offset = _unpack_search_string(payload, offset)
        best_move_uci, offset = _unpack_search_string(payload, offset)
        result_fen, offset = _unpack_search_string(payload, offset)
        results.append({
            "best_move": int(best_move),
            "policy": policy,
            "p_flip": float(p_flip),
            "value": float(value),
            "sigma_q": float(sigma_q),
            "hbar_eff": float(hbar_eff),
            "stop_reason": stop_reason,
            "iterations": int(iterations),
            "dup_rate": float(dup_rate),
            "max_pending": int(max_pending),
            "avg_vvalue": float(avg_vvalue),
            "best_move_uci": best_move_uci,
            "result_fen": result_fen,
            "result_history_hashes": history_hashes,
        })
    if offset != len(payload):
        raise ValueError("search response trailing bytes")
    if wrapper_kind == _SEARCH_RESP_SINGLE:
        return {"result": results[0] if results else {}}
    if wrapper_kind == _SEARCH_RESP_SESSION:
        return {"session_id": int(session_id), "results": results}
    if wrapper_kind == _SEARCH_RESP_MULTI:
        return {"results": results}
    raise ValueError(f"unknown search response wrapper kind: {wrapper_kind}")


__all__ = [
    "QIPC_BATCH_EVAL_REQ",
    "QIPC_BATCH_EVAL_REQ_SHM",
    "QIPC_BATCH_EVAL_RESP",
    "QIPC_BATCH_EVAL_RESP_SHM",
    "QIPC_EVAL_REQ",
    "QIPC_EVAL_REQ_SHM",
    "QIPC_EVAL_RESP",
    "QIPC_EVAL_RESP_SHM",
    "QIPC_HEADER",
    "QIPC_MAGIC",
    "QIPC_SHM_DEFAULT_BYTES",
    "QIPC_SHM_LEN",
    "QipcSharedMemoryTransport",
    "SHM_DIR_TO_PYTHON",
    "SHM_DIR_TO_RUST",
    "SHM_MSG_EVAL_BATCH_REQ",
    "SHM_MSG_EVAL_BATCH_RESP",
    "SHM_MSG_JSON",
    "SHM_MSG_SEARCH_RESP",
    "SHM_RING_DEFAULT_SIZE",
    "SHM_RING_HEADER_SIZE",
    "SHM_RING_MAGIC",
    "SHM_RING_SLOT_HEADER",
    "SHM_RING_VERSION",
    "SHM_SLOT_DONE",
    "SHM_SLOT_EMPTY",
    "SHM_SLOT_WRITTEN",
    "ShmRingBuffer",
    "cleanup_all_shm",
    "pack_qipc_batch_eval_resp",
    "pack_qipc_eval_resp",
    "register_ring_buffer",
    "unpack_qipc_batch_eval_req",
    "unpack_qipc_eval_req",
    "unpack_shm_search_response",
    "unregister_ring_buffer",
]
