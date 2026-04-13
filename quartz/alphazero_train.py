#!/usr/bin/env python3
"""
QUARTZ AlphaZero Training Pipeline — End-to-End
================================================
Actual working self-play → replay → train → checkpoint loop.

Usage:
  python3 -m quartz.train --game gomoku15 --iterations 50
  python3 -m quartz.train --game gomoku7 --iterations 10 --device cpu
  python3 -m quartz.train --serve --game gomoku15 --model alphazero_gomoku15/best.pt

Requirements: torch, numpy, tqdm
GPU:          pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
              export HSA_OVERRIDE_GFX_VERSION=10.3.0
"""
import os, sys, json, select, time, argparse, subprocess, random, math, signal, threading, logging, struct, warnings, atexit, queue
import numpy as np
from dataclasses import dataclass

log = logging.getLogger(__name__)
from pathlib import Path
from collections import OrderedDict, deque
from multiprocessing import shared_memory

from quartz.backend import (
    load_torch_state_dict,
    load_torch_state_dict_checked,
    validate_torch_state_dict,
)

try:
    import orjson
except ImportError:
    orjson = None

try:
    from quartz.gpu_detect import detect_gpu, GpuInfo
except ImportError:
    try:
        from gpu_detect import detect_gpu, GpuInfo
    except ImportError:
        detect_gpu = None
        GpuInfo = None

# Game-agnostic encoder system
try:
    from quartz.encoders import get_encoder, GameEncoder
except ImportError:
    # Fallback: encoders.py in same directory
    try:
        from encoders import get_encoder, GameEncoder
    except ImportError:
        get_encoder = None


def encode_board(cfg, board_flat, player):
    """Game-agnostic board encoding using registered encoder.
    For 17-channel history encoding, use _encode_board_with_history instead.
    This function creates a single-timestep snapshot (t=0 only, no history)."""
    enc_obj = cfg.get('_encoder')
    if enc_obj is not None:
        return enc_obj.encode(board_flat, player)
    bs = cfg['board']; n2 = bs * bs
    ch = cfg.get('ch', 17)
    enc = np.zeros((ch, bs, bs), dtype=np.float32)
    for i in range(n2):
        r, c = i // bs, i % bs
        if board_flat[i] == player: enc[0, r, c] = 1.0
        elif board_flat[i] != 0: enc[1, r, c] = 1.0
    # Color plane (last channel)
    if player == 1: enc[ch - 1] = 1.0
    return enc


def decode_board(cfg, enc, player):
    """Reconstruct flat board from encoded tensor."""
    enc_obj = cfg.get('_encoder')
    if enc_obj is not None:
        return enc_obj.decode(enc, player)
    # Legacy fallback
    bs = cfg['board']; board = np.zeros(bs * bs, dtype=np.int8)
    for r in range(bs):
        for c in range(bs):
            if enc[0, r, c] > 0.5: board[r * bs + c] = player
            elif enc[1, r, c] > 0.5: board[r * bs + c] = -player
    return board


def json_loads_fast(payload):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def json_dumps_compact(payload):
    if orjson is not None:
        out = orjson.dumps(payload)
        return out.decode("utf-8") if isinstance(out, bytes) else out
    return json.dumps(payload, separators=(",", ":"))


def iter_sparse_policy_entries(entries):
    for entry in entries or ():
        if isinstance(entry, str) and ":" in entry:
            idx_raw, val_raw = entry.split(":", 1)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            idx_raw, val_raw = entry[0], entry[1]
        else:
            continue
        try:
            idx = int(idx_raw)
            val = float(val_raw)
        except (TypeError, ValueError):
            continue
        yield idx, val


def dense_policy_from_sparse(entries, n_actions):
    policy = np.zeros(n_actions, dtype=np.float32)
    for idx, val in iter_sparse_policy_entries(entries):
        if 0 <= idx < n_actions:
            policy[idx] = val
    return policy


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


def _register_qipc_transport(transport):
    key = (transport.req.name, transport.resp.name)
    with _QIPC_TRANSPORTS_LOCK:
        _QIPC_TRANSPORTS[key] = transport


def _unregister_qipc_transport(transport):
    key = (transport.req.name, transport.resp.name)
    with _QIPC_TRANSPORTS_LOCK:
        _QIPC_TRANSPORTS.pop(key, None)


def _register_ring_buffer(ring):
    with _SHM_RING_BUFFERS_LOCK:
        _SHM_RING_BUFFERS[ring.name] = ring


def _unregister_ring_buffer(ring):
    with _SHM_RING_BUFFERS_LOCK:
        _SHM_RING_BUFFERS.pop(ring.name, None)


def _cleanup_all_shm():
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


atexit.register(_cleanup_all_shm)


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
        _register_qipc_transport(transport)
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
        _unregister_qipc_transport(self)

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


# ─── SHM Ring Buffer for lock-free eval pipeline ───

SHM_RING_MAGIC = 0x51524E47  # "QRNG"
SHM_RING_VERSION = 1
SHM_RING_HEADER_SIZE = 256
SHM_RING_SLOT_HEADER = 16
SHM_RING_DEFAULT_SIZE = 16 * 1024 * 1024  # 16MB

SHM_SLOT_EMPTY = 0
SHM_SLOT_WRITTEN = 1
SHM_SLOT_DONE = 2

SHM_MSG_EVAL_BATCH_REQ = 1
SHM_MSG_EVAL_BATCH_RESP = 2
SHM_MSG_JSON = 3
SHM_MSG_SEARCH_RESP = 4

SHM_DIR_TO_PYTHON = 0
SHM_DIR_TO_RUST = 1

import ctypes


@dataclass
class ShmRingBuffer:
    """Lock-free ring buffer in shared memory for Rust↔Python eval communication."""
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
        # Write header
        struct.pack_into("<IIIII", shm.buf, 0,
                         SHM_RING_MAGIC, SHM_RING_VERSION, r2p_slots, p2r_slots, slot_data_size)
        # Zero epoch, cmd_done
        struct.pack_into("<IB", shm.buf, 20, 0, 0)
        # Zero all slot states
        r2p_base = SHM_RING_HEADER_SIZE
        p2r_base = SHM_RING_HEADER_SIZE + r2p_slots * slot_data_size
        for i in range(total_slots):
            off = SHM_RING_HEADER_SIZE + i * slot_data_size
            shm.buf[off] = SHM_SLOT_EMPTY
        ring = cls(_shm=shm, r2p_slot_count=r2p_slots, p2r_slot_count=p2r_slots,
                   slot_data_size=slot_data_size, r2p_base=r2p_base, p2r_base=p2r_base)
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
        ring = cls(_shm=shm, r2p_slot_count=r2p_slots, p2r_slot_count=p2r_slots,
                   slot_data_size=slot_data_size, r2p_base=r2p_base, p2r_base=p2r_base)
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

    # --- Atomic accessors (x86-64: aligned byte/word loads/stores are atomic) ---

    def _atomic_load_u8(self, offset):
        return ctypes.c_uint8.from_buffer(self._shm.buf, offset).value

    def _atomic_store_u8(self, offset, val):
        ctypes.c_uint8.from_buffer(self._shm.buf, offset).value = val

    def _atomic_load_u32(self, offset):
        return ctypes.c_uint32.from_buffer(self._shm.buf, offset).value

    def _atomic_store_u32(self, offset, val):
        ctypes.c_uint32.from_buffer(self._shm.buf, offset).value = val

    # --- Header ---

    def epoch(self):
        return self._atomic_load_u32(20)

    def cmd_done(self):
        return self._atomic_load_u8(24) != 0

    def request_cancel(self):
        """Signal Rust to cancel the current command at the next wave boundary."""
        self._atomic_store_u8(25, 1)

    def cancel_requested(self):
        return self._atomic_load_u8(25) != 0

    # --- Slot state ---

    def _r2p_slot_offset(self, idx):
        return self.r2p_base + idx * self.slot_data_size

    def _p2r_slot_offset(self, idx):
        return self.p2r_base + idx * self.slot_data_size

    def slot_state(self, slot_offset):
        return self._atomic_load_u8(slot_offset)

    def set_slot_state(self, slot_offset, state):
        self._atomic_store_u8(slot_offset, state)

    # --- Read from r2p (Python reads Rust's messages) ---

    def r2p_try_read(self, slot_idx):
        """Try to read a WRITTEN r2p slot. Returns (msg_type, payload_bytes) or None."""
        off = self._r2p_slot_offset(slot_idx)
        if self.slot_state(off) != SHM_SLOT_WRITTEN:
            return None
        msg_type = self._shm.buf[off + 1]
        payload_len = struct.unpack_from("<I", self._shm.buf, off + 4)[0]
        payload = bytes(self._shm.buf[off + SHM_RING_SLOT_HEADER: off + SHM_RING_SLOT_HEADER + payload_len])
        return msg_type, payload

    def r2p_try_read_meta(self, slot_idx):
        """Try to read a WRITTEN r2p slot with metadata. Returns (msg_type, epoch, seq, payload) or None."""
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
        """Mark an r2p slot as DONE (processed by Python)."""
        off = self._r2p_slot_offset(slot_idx)
        self.set_slot_state(off, SHM_SLOT_DONE)

    # --- Write to p2r (Python writes responses to Rust) ---

    def p2r_try_write(self, slot_idx, msg_type, payload, epoch=0, seq=0):
        """Write to a p2r slot. Caller must ensure slot is EMPTY."""
        off = self._p2r_slot_offset(slot_idx)
        if len(payload) > self.slot_data_size - SHM_RING_SLOT_HEADER:
            return False
        # Write metadata
        self._shm.buf[off + 1] = msg_type
        self._shm.buf[off + 2] = SHM_DIR_TO_RUST
        self._shm.buf[off + 3] = 0
        struct.pack_into("<III", self._shm.buf, off + 4, len(payload), epoch, seq)
        # Write payload
        self._shm.buf[off + SHM_RING_SLOT_HEADER: off + SHM_RING_SLOT_HEADER + len(payload)] = payload
        # Set state to WRITTEN (release)
        self.set_slot_state(off, SHM_SLOT_WRITTEN)
        return True

    def p2r_slot_state(self, slot_idx):
        return self.slot_state(self._p2r_slot_offset(slot_idx))

    def slot_payload_capacity(self):
        return self.slot_data_size - SHM_RING_SLOT_HEADER


def _get_qipc_transport(proc):
    return getattr(proc, "_quartz_qipc_transport", None)


def _cleanup_qipc_transport(proc):
    transport = _get_qipc_transport(proc)
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
            _unregister_ring_buffer(ring)
        finally:
            try:
                delattr(proc, "_quartz_ring_buffer")
            except Exception:
                pass


def _json_line_bytes(payload):
    if isinstance(payload, (bytes, bytearray)):
        out = bytes(payload)
    else:
        out = json_dumps_compact(payload).encode("utf-8")
    return out if out.endswith(b"\n") else out + b"\n"


def _read_exact(stream, n_bytes, timeout_s=None):
    chunks = bytearray()
    deadline = None if timeout_s is None else time.perf_counter() + float(timeout_s)
    while len(chunks) < n_bytes:
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0 or not wait_readable(stream, remaining):
                raise TimeoutError(f"timed out reading {n_bytes} bytes from IPC stream")
        chunk = stream.read(n_bytes - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _stall_trace_path():
    path = os.environ.get("QUARTZ_STALL_TRACE_PATH", "").strip()
    return path or None


def _stall_trace(event, **fields):
    path = _stall_trace_path()
    if not path:
        return
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "event": str(event),
    }
    for key, value in fields.items():
        if isinstance(value, (np.generic,)):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        record[str(key)] = value
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass


def proc_write_json_line(proc_or_stream, payload):
    stream = getattr(proc_or_stream, "stdin", proc_or_stream)
    stream.write(_json_line_bytes(payload))
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


def proc_read_message(proc_or_stream, timeout_s=None):
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
            return "json", json_loads_fast(text)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("proc_read_message: JSON parse failed (%s), skipping line", exc)
            return "json", None
    header_rest = _read_exact(stream, QIPC_HEADER.size - 1, timeout_s=timeout_s)
    if header_rest is None:
        return None, None
    try:
        magic, frame_kind, payload_len = QIPC_HEADER.unpack(first + header_rest)
    except struct.error as exc:
        log.warning("proc_read_message: QIPC header unpack failed (%s), skipping", exc)
        return None, None
    if magic != QIPC_MAGIC:
        log.warning("proc_read_message: unexpected IPC frame magic: %r", magic)
        return None, None
    if payload_len > 256 * 1024 * 1024:  # 256 MB sanity cap
        log.warning("proc_read_message: unreasonable payload_len=%d, skipping", payload_len)
        return None, None
    payload = _read_exact(stream, payload_len, timeout_s=timeout_s)
    if payload is None:
        return None, None
    return "frame", (frame_kind, payload)


def proc_decode_eval_frame(proc, frame_kind, payload):
    transport = _get_qipc_transport(proc)
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
    transport = _get_qipc_transport(proc)
    if prefer_shm and transport is not None and transport.write_response(payload):
        shm_kind = {
            QIPC_EVAL_RESP: QIPC_EVAL_RESP_SHM,
            QIPC_BATCH_EVAL_RESP: QIPC_BATCH_EVAL_RESP_SHM,
        }.get(logical_kind)
        if shm_kind is not None:
            proc_write_qipc_frame(proc, shm_kind, QIPC_SHM_LEN.pack(len(payload)))
            return
    proc_write_qipc_frame(proc, logical_kind, payload)


def unpack_qipc_eval_req(payload):
    if len(payload) < 8:
        raise ValueError("short eval_req payload")
    if len(payload) >= 12:
        model_tag, num_actions, feat_len = struct.unpack_from("<III", payload, 0)
        offset = 12
    else:
        model_tag = 0
        num_actions, feat_len = struct.unpack_from("<II", payload, 0)
        offset = 8
    expected_bytes = offset + feat_len * 4
    if len(payload) != expected_bytes:
        raise ValueError("eval_req payload length mismatch")
    features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=offset)
    return num_actions, features, int(model_tag)


def unpack_qipc_batch_eval_req(payload):
    if len(payload) < 4:
        raise ValueError("short batch_eval_req payload")
    batch_size, = struct.unpack_from("<I", payload, 0)
    offset = 4
    requests = []
    for _ in range(batch_size):
        if offset + 12 <= len(payload):
            model_tag, num_actions, feat_len = struct.unpack_from("<III", payload, offset)
            offset += 12
        elif offset + 8 <= len(payload):
            model_tag = 0
            num_actions, feat_len = struct.unpack_from("<II", payload, offset)
            offset += 8
        else:
            raise ValueError("truncated batch_eval_req header")
        byte_len = feat_len * 4
        if offset + byte_len > len(payload):
            raise ValueError("truncated batch_eval_req features")
        features = np.frombuffer(payload, dtype="<f4", count=feat_len, offset=offset)
        offset += byte_len
        requests.append((num_actions, features, int(model_tag)))
    if offset != len(payload):
        raise ValueError("batch_eval_req trailing bytes")
    return requests


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
        result = {
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
        }
        results.append(result)
    if offset != len(payload):
        raise ValueError("search response trailing bytes")
    if wrapper_kind == _SEARCH_RESP_SINGLE:
        return {"result": results[0] if results else {}}
    if wrapper_kind == _SEARCH_RESP_SESSION:
        return {"session_id": int(session_id), "results": results}
    if wrapper_kind == _SEARCH_RESP_MULTI:
        return {"results": results}
    raise ValueError(f"unknown search response wrapper kind: {wrapper_kind}")


class InferencePipelineThread:
    """Background thread that runs GPU inference while the caller collects the next batch.

    Usage:
        pipeline = InferencePipelineThread(model, device, cfg)
        pipeline.start()
        pipeline.submit(eval_groups_0)
        # ... collect eval_groups_1 while inference_0 runs ...
        results_0 = pipeline.collect()
        pipeline.submit(eval_groups_1)
        # ...
        pipeline.stop()

    Torch-only: the GIL is released during CUDA kernel execution, allowing the
    collector thread to run Python (QIPC parsing) in parallel with GPU inference.
    """

    def __init__(self, model, device, cfg, max_pending=1):
        self._model = model
        self._device = device
        self._cfg = cfg
        self._inbound = queue.Queue(maxsize=max_pending)
        self._outbound = queue.Queue(maxsize=max_pending)
        self._shutdown = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="quartz-inference")
        self._thread.start()

    def stop(self, timeout=5.0):
        self._shutdown.set()
        try:
            self._inbound.put_nowait(None)  # sentinel to unblock
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, eval_groups):
        """Submit a batch for background inference. Blocks if pipeline is full."""
        self._inbound.put(eval_groups, timeout=10.0)

    def collect(self, timeout=30.0):
        """Block until the oldest submitted batch completes. Re-raises inference errors."""
        result = self._outbound.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise result
        return result

    def _loop(self):
        while not self._shutdown.is_set():
            try:
                groups = self._inbound.get(timeout=0.5)
            except queue.Empty:
                continue
            if groups is None:  # shutdown sentinel
                break
            try:
                responses = _run_batched_eval_groups(
                    groups, self._model, self._device, self._cfg)
                self._outbound.put(responses)
            except Exception as e:
                self._outbound.put(e)


class NNEvalCache:
    """LRU cache for NN evaluation results keyed by feature hash."""

    def __init__(self, max_entries=65536):
        self._cache = OrderedDict()  # hash → (policy_np, value_float)
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, feat_hash):
        with self._lock:
            entry = self._cache.get(feat_hash)
            if entry is None:
                self._misses += 1
                return None
            try:
                self._cache.move_to_end(feat_hash)
            except KeyError:
                # A concurrent clear/eviction should degrade to a miss, not crash
                # the self-play worker hot path.
                self._misses += 1
                return None
            self._hits += 1
            return entry

    def put(self, feat_hash, policy_np, value):
        with self._lock:
            if feat_hash in self._cache:
                self._cache.move_to_end(feat_hash)
            self._cache[feat_hash] = (policy_np, float(value))
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def hit_rate(self):
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @staticmethod
    def default_size(actions):
        """Size scaled by action space: larger actions → fewer entries to fit in ~256MB."""
        entry_bytes = actions * 4 + 8
        return min(131072, max(4096, 256 * 1024 * 1024 // entry_bytes))


_NN_EVAL_CACHE = None  # lazily initialized per-model


def _get_nn_eval_cache(cfg):
    global _NN_EVAL_CACHE
    if os.environ.get("QUARTZ_DISABLE_NN_CACHE"):
        return None
    if _NN_EVAL_CACHE is None:
        actions = cfg.get("actions", 49)
        _NN_EVAL_CACHE = NNEvalCache(NNEvalCache.default_size(actions))
    return _NN_EVAL_CACHE


def clear_nn_eval_cache():
    """Call after model training to invalidate cached predictions."""
    global _NN_EVAL_CACHE
    if _NN_EVAL_CACHE is not None:
        if _NN_EVAL_CACHE._hits + _NN_EVAL_CACHE._misses > 0:
            log.info("NN cache: hit_rate=%.1f%% entries=%d",
                     _NN_EVAL_CACHE.hit_rate * 100, len(_NN_EVAL_CACHE._cache))
        _NN_EVAL_CACHE.clear()


def _run_batched_eval_groups(eval_groups, model, device, cfg):
    if not eval_groups:
        return []
    ch_cfg, bs_cfg = cfg["ch"], cfg["board"]
    expected = ch_cfg * bs_cfg * bs_cfg
    nn_cache = _get_nn_eval_cache(cfg)
    flat_requests = []
    batch_features = []
    cached_results = {}  # index → (policy, value)
    for group in eval_groups:
        for request in group["requests"]:
            if len(request) == 3:
                na, feats, model_tag = request
            else:
                na, feats = request
                model_tag = 0
            na = max(1, int(na))
            idx = len(flat_requests)
            flat_requests.append((na, int(model_tag)))
            if len(feats) == expected:
                x = np.asarray(feats, dtype=np.float32).reshape(ch_cfg, bs_cfg, bs_cfg)
            else:
                x = np.zeros((ch_cfg, bs_cfg, bs_cfg), dtype=np.float32)
            # Check cache (include model_tag to avoid cross-model pollution)
            if nn_cache is not None:
                feat_hash = hash((int(model_tag), x.tobytes()))
                hit = nn_cache.get(feat_hash)
                if hit is not None:
                    cached_results[idx] = hit
                    batch_features.append(None)  # placeholder
                    continue
            batch_features.append(x)

    if isinstance(model, dict):
        model_map = {int(k): v for k, v in model.items()}
    else:
        model_map = None

    # Fill cached results first, build GPU batch from misses only
    all_policies = [None] * len(flat_requests)
    all_values = [0.0] * len(flat_requests)
    for idx, (policy, value) in cached_results.items():
        na = flat_requests[idx][0]
        all_policies[idx] = policy[:na] if len(policy) >= na else policy
        all_values[idx] = value

    # Filter to GPU-needed items only
    gpu_indices = [i for i in range(len(flat_requests)) if i not in cached_results]
    gpu_features = [batch_features[i] for i in gpu_indices if batch_features[i] is not None]

    if gpu_features:
        if model_map is not None:
            features_np = np.stack(gpu_features, axis=0)
            by_tag = {}
            for local_i, global_i in enumerate(gpu_indices):
                na, model_tag = flat_requests[global_i]
                by_tag.setdefault(int(model_tag), []).append((local_i, global_i, int(na)))
            for model_tag, entries in by_tag.items():
                model_obj = model_map.get(int(model_tag))
                local_idxs = [li for li, _, _ in entries]
                if model_obj is not None:
                    probs_batch, vals_np = _run_model_batch(model_obj, device, features_np[local_idxs])
                    for bi, (_, global_i, na) in enumerate(entries):
                        all_policies[global_i] = probs_batch[bi][:na]
                        all_values[global_i] = float(vals_np[bi])
                else:
                    for _, global_i, na in entries:
                        all_policies[global_i] = np.full(na, 1.0 / na, dtype=np.float32)
                        all_values[global_i] = 0.0
        elif model is not None:
            probs_batch, vals_np = _run_model_batch(model, device, np.stack(gpu_features, axis=0))
            for bi, global_i in enumerate(gpu_indices):
                na = flat_requests[global_i][0]
                all_policies[global_i] = probs_batch[bi][:na]
                all_values[global_i] = float(vals_np[bi])
        else:
            for global_i in gpu_indices:
                na = flat_requests[global_i][0]
                all_policies[global_i] = np.full(na, 1.0 / na, dtype=np.float32)
                all_values[global_i] = 0.0
    # Fill any remaining None entries (items with no features and no cache)
    for i in range(len(all_policies)):
        if all_policies[i] is None:
            na = flat_requests[i][0]
            all_policies[i] = np.full(na, 1.0 / na, dtype=np.float32)

    # Store GPU results in cache
    if nn_cache is not None:
        for global_i in gpu_indices:
            feat = batch_features[global_i]
            if feat is not None:
                model_tag_i = flat_requests[global_i][1]
                feat_hash = hash((model_tag_i, feat.tobytes()))
                nn_cache.put(feat_hash, all_policies[global_i], all_values[global_i])

    responses = []
    offset = 0
    for group in eval_groups:
        count = len(group["requests"])
        responses.append({
            # Some older / alternate parse paths may omit gi; default to 0 so
            # batched evaluation does not crash on a shape mismatch.
            "gi": int(group.get("gi", 0)),
            "kind": group["kind"],
            "policies": all_policies[offset:offset + count],
            "values": all_values[offset:offset + count],
        })
        offset += count
    return responses


def _make_eval_request_group(kind, requests, gi=0, prefer_shm=False):
    normalized = []
    for req in requests:
        if len(req) == 3:
            na, feats, model_tag = req
        else:
            na, feats = req
            model_tag = 0
        normalized.append((max(1, int(na)), feats, int(model_tag)))
    group = {
        "gi": int(gi),
        "kind": kind,
        "requests": normalized,
    }
    if prefer_shm:
        group["prefer_shm"] = True
    return group


def _write_batched_eval_group(proc, response_group):
    kind = response_group["kind"]
    policies = response_group["policies"]
    values = response_group["values"]
    if kind == "binary_batch":
        proc_write_eval_response(
            proc,
            QIPC_BATCH_EVAL_RESP,
            pack_qipc_batch_eval_resp(policies, values),
            prefer_shm=bool(response_group.get("prefer_shm")),
        )
    elif kind == "binary_single":
        proc_write_eval_response(
            proc,
            QIPC_EVAL_RESP,
            pack_qipc_eval_resp(policies[0], values[0]),
            prefer_shm=bool(response_group.get("prefer_shm")),
        )
    elif kind == "json_batch":
        proc_write_json_line(proc, {
            "batch_eval_resp": {
                "responses": [
                    {"policy": policy.tolist(), "value": float(value)}
                    for policy, value in zip(policies, values)
                ]
            }
        })
    elif kind == "json_single":
        proc_write_json_line(proc, {
            "eval_resp": {
                "policy": policies[0].tolist(),
                "value": float(values[0]),
            }
        })
    else:
        raise ValueError(f"unknown eval response group kind: {kind}")


def _shm_eval_loop(ring, model, device, cfg, proc, on_json=None):
    """SHM ring buffer eval loop.

    Reads eval batch requests and JSON messages from r2p slots,
    writes eval batch responses to p2r slots.
    Uses InferencePipelineThread for GPU overlap when possible.

    Returns when Rust sets cmd_done flag. If the command writes a binary
    search response onto the ring, that decoded payload is returned.

    Args:
        ring: ShmRingBuffer instance
        model: PyTorch model (or dict of models)
        device: torch device
        cfg: game config dict
        proc: subprocess with stdin/stdout
        on_json: callback(dict) for JSON messages (selfplay_chunk, selfplay_progress, etc.)
    """
    _use_pipeline = (
        not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
        and model is not None
        and not hasattr(model, "predict")
    )
    pipeline = None
    inflight = False
    _inflight_epoch = 0
    _inflight_seq = 0
    _duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
    _duty_log_interval = 16

    if _use_pipeline:
        pipeline = InferencePipelineThread(model, device, cfg, max_pending=1)
        pipeline.start()

    terminal_payload = None
    try:
        spin = 0
        while True:
            # --- Flush inflight inference before reading next batch ---
            if inflight and pipeline is not None:
                model_t0 = time.perf_counter()
                responses = pipeline.collect(timeout=30.0)
                _duty["model_s"] += time.perf_counter() - model_t0
                inflight = False
                write_t0 = time.perf_counter()
                for rg in responses:
                    _shm_write_eval_response(ring, rg, epoch=_inflight_epoch, seq=_inflight_seq)
                _duty["write_s"] += time.perf_counter() - write_t0

            # --- Poll all r2p slots, process each immediately ---
            read_t0 = time.perf_counter()
            found_eval = False
            for slot_idx in range(ring.r2p_slot_count):
                result = ring.r2p_try_read_meta(slot_idx)
                if result is None:
                    continue
                msg_type, req_epoch, req_seq, payload_bytes = result
                spin = 0

                if msg_type == SHM_MSG_EVAL_BATCH_REQ:
                    ring.r2p_mark_done(slot_idx)
                    _duty["read_s"] += time.perf_counter() - read_t0
                    found_eval = True
                    requests = unpack_qipc_batch_eval_req(bytes(payload_bytes))
                    eval_groups = [_make_eval_request_group("binary_batch", requests, gi=0)]

                    if pipeline is not None:
                        _inflight_epoch = req_epoch
                        _inflight_seq = req_seq
                        pipeline.submit(eval_groups)
                        inflight = True
                    else:
                        model_t0 = time.perf_counter()
                        responses = _run_batched_eval_groups(eval_groups, model, device, cfg)
                        _duty["model_s"] += time.perf_counter() - model_t0
                        write_t0 = time.perf_counter()
                        for rg in responses:
                            _shm_write_eval_response(ring, rg, epoch=req_epoch, seq=req_seq)
                        _duty["write_s"] += time.perf_counter() - write_t0

                    _duty["cycles"] += 1
                    if _duty["cycles"] % _duty_log_interval == 0:
                        NNSearchClient._emit_duty_cycle(_duty)
                    # Break to flush pipeline before reading next batch
                    break

                elif msg_type == SHM_MSG_JSON:
                    ring.r2p_mark_done(slot_idx)
                    try:
                        json_obj = json_loads_fast(payload_bytes.decode("utf-8"))
                        if callable(on_json) and json_obj:
                            on_json(json_obj)
                    except Exception:
                        pass
                elif msg_type == SHM_MSG_SEARCH_RESP:
                    ring.r2p_mark_done(slot_idx)
                    try:
                        terminal_payload = unpack_shm_search_response(payload_bytes)
                    except Exception:
                        terminal_payload = {"error": "invalid shm search response"}
                else:
                    ring.r2p_mark_done(slot_idx)

            if not found_eval:
                _duty["read_s"] += time.perf_counter() - read_t0
                if ring.cmd_done():
                    # Drain any remaining ring messages for this command.
                    for slot_idx in range(ring.r2p_slot_count):
                        result = ring.r2p_try_read(slot_idx)
                        if result is None:
                            continue
                        msg_type, payload_bytes = result
                        ring.r2p_mark_done(slot_idx)
                        if msg_type == SHM_MSG_JSON:
                            try:
                                json_obj = json_loads_fast(payload_bytes.decode("utf-8"))
                                if callable(on_json) and json_obj:
                                    on_json(json_obj)
                            except Exception:
                                pass
                        elif msg_type == SHM_MSG_SEARCH_RESP:
                            try:
                                terminal_payload = unpack_shm_search_response(payload_bytes)
                            except Exception:
                                terminal_payload = {"error": "invalid shm search response"}
                    break

                # Check if the Rust process died (killed by pause or crash)
                if proc.poll() is not None:
                    raise RuntimeError(f"Rust server exited (code={proc.returncode}) during SHM eval loop")

                spin += 1
                if spin < 64:
                    pass  # busy spin
                elif spin < 512:
                    time.sleep(0.000001)  # 1µs
                else:
                    time.sleep(0.00001)   # 10µs

    finally:
        if inflight and pipeline is not None:
            try:
                drain = pipeline.collect(timeout=10.0)
                for rg in drain:
                    _shm_write_eval_response(ring, rg, epoch=_inflight_epoch, seq=_inflight_seq)
            except Exception:
                pass
        if pipeline is not None:
            pipeline.stop()
        if _duty["cycles"] > 0:
            NNSearchClient._emit_duty_cycle(_duty)
    return terminal_payload


def _shm_write_eval_response(ring, response_group, epoch=0, seq=0):
    """Write eval response to p2r ring buffer slot with matching epoch/seq."""
    policies = response_group["policies"]
    values = response_group["values"]
    payload = pack_qipc_batch_eval_resp(policies, values)
    # Spin-wait for empty p2r slot
    for attempt in range(100000):
        for slot_idx in range(ring.p2r_slot_count):
            if ring.p2r_try_write(slot_idx, SHM_MSG_EVAL_BATCH_RESP, payload, epoch=epoch, seq=seq):
                return
        if attempt < 64:
            pass
        elif attempt < 512:
            time.sleep(0.000001)
        else:
            time.sleep(0.00001)
    log.warning("_shm_write_eval_response: timed out waiting for p2r slot")


def launch_rust_server(rust_binary):
    """Start Rust server and fail fast with stderr if it exits immediately."""
    transport = None
    ring_buffer = None
    env = os.environ.copy()
    disable_shm = str(env.get("QUARTZ_DISABLE_QIPC_SHM", "")).strip().lower() in {"1", "true", "yes", "on"}
    try:
        if not disable_shm:
            transport = QipcSharedMemoryTransport.create()
            env["QUARTZ_QIPC_REQ_SHM_NAME"] = transport.req.name
            env["QUARTZ_QIPC_RESP_SHM_NAME"] = transport.resp.name
            env["QUARTZ_QIPC_REQ_SHM_SIZE"] = str(transport.size)
            env["QUARTZ_QIPC_RESP_SHM_SIZE"] = str(transport.size)
            # Also create SHM ring buffer for lock-free eval pipeline
            try:
                ring_buffer = ShmRingBuffer.create(r2p_slots=2, p2r_slots=2)
                env["QUARTZ_QIPC_RING_SHM_NAME"] = ring_buffer.name
                env["QUARTZ_QIPC_RING_SHM_SIZE"] = str(ring_buffer.size)
                _register_ring_buffer(ring_buffer)
            except Exception:
                ring_buffer = None
    except Exception:
        transport = None
    _stall_trace("rust_server_launch", rust_binary=rust_binary, shm=bool(transport))
    proc = subprocess.Popen(
        [rust_binary, "--server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=False, bufsize=0, env=env)
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
    _stall_trace("rust_server_ready", child_pid=proc.pid, shm=bool(transport), ring=bool(ring_buffer))
    return proc


def stop_rust_server(proc, timeout=3.0):
    if proc is None:
        return
    _stall_trace("rust_server_stop_begin", child_pid=getattr(proc, "pid", None))
    try:
        proc_write_json_line(proc, {"cmd": "quit"})
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _cleanup_qipc_transport(proc)
        _stall_trace("rust_server_stop_end", child_pid=getattr(proc, "pid", None), returncode=getattr(proc, "returncode", None))


class RustServerPool:
    """Small reusable pool of Rust search servers for batched self-play."""

    def __init__(self, rust_binary):
        self.rust_binary = rust_binary
        self._lock = threading.Lock()
        self._procs = []

    def acquire(self, n):
        with self._lock:
            alive = []
            for proc in self._procs:
                if proc.poll() is None:
                    alive.append(proc)
                else:
                    stop_rust_server(proc, timeout=0.1)
            self._procs = alive
            while len(self._procs) < n:
                self._procs.append(launch_rust_server(self.rust_binary))
            return list(self._procs[:n])

    def kill_active(self):
        """Kill all active processes to force in-flight operations to abort.
        Pool will spawn fresh processes on next acquire()."""
        with self._lock:
            for proc in self._procs:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._procs = []

    def close(self):
        with self._lock:
            procs, self._procs = self._procs, []
        for proc in procs:
            stop_rust_server(proc)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        """Fallback when tqdm not installed."""
        def __init__(self, iterable=None, total=None, desc="", leave=True, **kw):
            self.iterable = iterable; self.total = total; self.desc = desc; self.n = 0
        def __iter__(self):
            for x in self.iterable: yield x; self.n += 1
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        def set_postfix_str(self, s): pass
        def set_postfix(self, **kw): pass
        def close(self): pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CHESS_POLICY_ACTIONS = 4672
SEARCH_RUNTIME_KEYS = {
    "sigma_0",
    "min_visits",
    "check_interval",
    "prior_refresh_rate",
    "prior_refresh_temp",
    "hbar_penalty_cap",
    "c_puct",
    "penalty_mode",
    "root_only_shaping",
    "n_threads",
    "batch_size",
    "batch_timeout_us",
}

GOMOKU15_VARIANTS = {
    "gomoku15",
    "gomoku15_free",
    "gomoku15_std",
    "gomoku15_omok",
    "gomoku15_renju",
    "gomoku15_caro",
}

GO_RULESET_PRESETS = {
    "cn": dict(go_ruleset="chinese", go_scoring="area", go_komi=7.5, go_allow_suicide=False),
    "jp": dict(go_ruleset="japanese", go_scoring="territory", go_komi=6.5, go_allow_suicide=False),
    "kr": dict(go_ruleset="korean", go_scoring="territory", go_komi=6.5, go_allow_suicide=False),
}

STANDARD_CHESS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _make_go_cfg(board, filters, blocks, vh, iters, games, temp_th, dir_a, steps,
                 batch, min_visits, check_interval, recent_window, suffix="cn"):
    cfg = dict(
        board=board, ch=17, actions=board * board + 1, win=0,
        filters=filters, blocks=blocks, vh=vh, iters=iters, games=games,
        temp_th=temp_th, dir_a=dir_a, buf=1_000_000, steps=steps, batch=batch,
        penalty_mode="GatedRefresh", hbar_penalty_cap=0.3, sigma_0=0.3,
        min_visits=min_visits, check_interval=check_interval,
        prior_refresh_rate=0.0, prior_refresh_temp=1.0, c_puct=2.5,
        n_threads=4, batch_size=8, recent_frac=0.8, recent_window=recent_window,
        tt_enabled=True,
    )
    cfg.update(GO_RULESET_PRESETS[suffix])
    return cfg

# ═══════════════════════════════════════════
# § Game Configs
# ═══════════════════════════════════════════

_GOMOKU15_BASE = dict(
    board=15, ch=17, actions=225, win=5, filters=128, blocks=8, vh=256,
    iters=200, games=100, temp_th=15, dir_a=0.15, buf=500_000, steps=200,
    batch=512, penalty_mode="GatedRefresh", hbar_penalty_cap=0.3, sigma_0=0.3,
    min_visits=50, check_interval=100, prior_refresh_rate=0.0,
    prior_refresh_temp=1.0, c_puct=2.0, n_threads=4, batch_size=8,
    recent_frac=0.8, recent_window=50_000)

_CHESS_BASE = dict(
    board=8, ch=36, actions=CHESS_POLICY_ACTIONS, win=0, filters=128, blocks=10, vh=256,
    iters=800, games=40, temp_th=15, dir_a=0.3, buf=1_000_000, steps=400, batch=256,
    penalty_mode="GatedRefresh", hbar_penalty_cap=0.3, sigma_0=0.3, min_visits=50,
    check_interval=100, prior_refresh_rate=0.0, prior_refresh_temp=1.0, c_puct=2.5,
    n_threads=4, batch_size=8, recent_frac=0.8, recent_window=100_000,
    tt_enabled=True)

GAME_CONFIGS = {
    "gomoku7":  dict(board=7,  ch=17, actions=49,  win=4, filters=64,  blocks=4, vh=64,  iters=200, games=200, temp_th=8,  dir_a=0.5,  buf=200_000,  steps=100, batch=256, penalty_mode="GatedRefresh", hbar_penalty_cap=0.3, sigma_0=0.3, min_visits=15, check_interval=20, prior_refresh_rate=0.0, prior_refresh_temp=1.0, c_puct=2.0, n_threads=4, batch_size=8, recent_frac=0.8, recent_window=20_000),
    "gomoku15": dict(_GOMOKU15_BASE),
    "gomoku15_free": dict(_GOMOKU15_BASE),
    "gomoku15_std": dict(_GOMOKU15_BASE),
    "gomoku15_omok": dict(_GOMOKU15_BASE),
    "gomoku15_renju": dict(_GOMOKU15_BASE),
    "gomoku15_caro": dict(_GOMOKU15_BASE),
    "go9":      _make_go_cfg(9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "cn"),
    "go9_cn":   _make_go_cfg(9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "cn"),
    "go9_jp":   _make_go_cfg(9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "jp"),
    "go9_kr":   _make_go_cfg(9, 128, 10, 256, 600, 60, 12, 0.3, 300, 512, 50, 100, 100_000, "kr"),
    "go13":     _make_go_cfg(13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "cn"),
    "go13_cn":  _make_go_cfg(13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "cn"),
    "go13_jp":  _make_go_cfg(13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "jp"),
    "go13_kr":  _make_go_cfg(13, 160, 11, 320, 700, 32, 16, 0.28, 360, 448, 64, 128, 125_000, "kr"),
    "go19":     _make_go_cfg(19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "cn"),
    "go19_cn":  _make_go_cfg(19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "cn"),
    "go19_jp":  _make_go_cfg(19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "jp"),
    "go19_kr":  _make_go_cfg(19, 192, 12, 384, 800, 20, 20, 0.25, 400, 384, 80, 160, 150_000, "kr"),
    "chess":    dict(_CHESS_BASE, chess960=False, chess960_index=None),
    "chess960": dict(_CHESS_BASE, chess960=True, chess960_index=None),
    "tictactoe": dict(board=3, ch=17, actions=9, win=3, filters=32, blocks=2, vh=32, iters=100, games=400, temp_th=4, dir_a=0.6, buf=50_000, steps=64, batch=128, penalty_mode="GatedRefresh", hbar_penalty_cap=0.3, sigma_0=0.3, min_visits=8, check_interval=16, prior_refresh_rate=0.0, prior_refresh_temp=1.0, c_puct=1.5, n_threads=2, batch_size=8, recent_frac=0.8, recent_window=10_000),
}


def rust_game_name(game_name):
    if game_name in GAME_CONFIGS:
        return game_name
    if game_name in GOMOKU15_VARIANTS:
        return game_name
    return "gomoku15"


def is_chess_game(game_name):
    return game_name in {"chess", "chess960"}


def is_go_game(game_name):
    return bool(game_name) and game_name.startswith("go") and len(game_name) > 2 and game_name[2].isdigit()


def apply_config_overrides(cfg, overrides):
    merged = dict(cfg)
    unknown = []
    for key, value in overrides.items():
        if key in merged or key in SEARCH_RUNTIME_KEYS:
            merged[key] = value
        else:
            unknown.append(key)
    if unknown:
        print(f"  [WARN] Ignoring unsupported config keys: {', '.join(sorted(unknown))}")
    return merged


def resolve_runtime_paths(base_dir, explicit_model=None, resume=False):
    latest_model_path = os.path.join(base_dir, "latest.pt")
    best_model_path = os.path.join(base_dir, "best.pt")
    if explicit_model:
        load_model_path = explicit_model
    elif resume and os.path.exists(latest_model_path):
        load_model_path = latest_model_path
    elif resume and os.path.exists(best_model_path):
        load_model_path = best_model_path
    else:
        load_model_path = latest_model_path
    return {
        "load_model_path": load_model_path,
        "latest_model_path": latest_model_path,
        "best_model_path": best_model_path,
        "replay_path": os.path.join(base_dir, "replay.npz"),
        "log_path": os.path.join(base_dir, "train_log.jsonl"),
        "autotune_profile_path": os.path.join(base_dir, "autotune_profile.json"),
    }


def detect_checkpoint_backend_hint(path):
    """Best-effort checkpoint backend hint from local file header."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return "missing"
    if not head:
        return "empty"
    if head.startswith(b"PK\x03\x04"):
        return "torch"
    if b"jax._src" in head or b"flax" in head:
        return "jax"
    if head.startswith(b"\x80\x04") and b"params" in head and b"BatchNorm_" in head:
        return "jax"
    return "unknown"


def ensure_best_checkpoint_compatible(best_model_path, backend, model, device):
    """Keep best.pt aligned with the active training backend.

    Old experiments may leave a JAX checkpoint behind while current training is
    running with PyTorch. That breaks evaluation promotion loading. When the
    format obviously mismatches, reseed best.pt from the current model.
    """
    if not os.path.exists(best_model_path):
        return None
    active_backend = getattr(backend, "name", "torch") if backend is not None else "torch"
    hint = detect_checkpoint_backend_hint(best_model_path)
    mismatch = (
        (active_backend == "torch" and hint == "jax")
        or (active_backend == "jax" and hint == "torch")
    )
    if not mismatch and active_backend == "torch" and model is not None and hint in {"torch", "unknown"}:
        try:
            state_dict = load_torch_state_dict(best_model_path, torch, map_location=device)
            mismatch = validate_torch_state_dict(model, state_dict) is not None
        except Exception:
            mismatch = True
    if not mismatch:
        return hint
    if backend is not None:
        backend.save(best_model_path)
    elif model is not None:
        torch.save(model.state_dict(), best_model_path)
    print(f"  [Eval] Reset incompatible best checkpoint ({hint} -> {active_backend})")
    return active_backend


def rust_search_options(cfg, penalty_mode=None):
    n_threads = int(cfg.get("n_threads", 1) or 1)
    batch_size = int(cfg.get("batch_size", 8) or 8)
    batch_timeout_us = int(cfg.get(
        "batch_timeout_us",
        1500 if n_threads <= 1 else min(6000, 1200 + 250 * max(n_threads, batch_size // 2)),
    ) or 1500)
    return {
        "search_profile": cfg.get("search_profile", "quartz"),
        "penalty_mode": penalty_mode or cfg.get("penalty_mode", "GatedRefresh"),
        "hbar_penalty_cap": cfg.get("hbar_penalty_cap", 0.3),
        "sigma_0": cfg.get("sigma_0", 0.3),
        "min_visits": cfg.get("min_visits", 50),
        "check_interval": cfg.get("check_interval", 100),
        "prior_refresh_rate": cfg.get("prior_refresh_rate", 0.0),
        "prior_refresh_temp": cfg.get("prior_refresh_temp", 1.0),
        "c_puct": cfg.get("c_puct", 0.0),
        "n_threads": n_threads,
        "batch_size": batch_size,
        "batch_timeout_us": batch_timeout_us,
        **({"root_only_shaping": bool(cfg["root_only_shaping"])} if "root_only_shaping" in cfg else {}),
        **({"vl_mode": cfg["vl_mode"]} if "vl_mode" in cfg else {}),
        **({"tt_enabled": bool(cfg["tt_enabled"])} if "tt_enabled" in cfg else {}),
    }


def normalize_rust_board(game_name, board_flat):
    if board_flat is None:
        return None
    if is_go_game(game_name):
        return [1 if v == 1 else 2 if v in (-1, 2) else 0 for v in board_flat]
    return board_flat.tolist() if hasattr(board_flat, "tolist") else list(board_flat)


def chess_state_meta_from_hashes(history_hashes):
    hashes = []
    for value in history_hashes or []:
        try:
            hashes.append(int(value))
        except (TypeError, ValueError):
            continue
    return {"chess_history_hashes": hashes} if hashes else {}


def build_rust_state_meta(game_name, state, cfg):
    if is_chess_game(game_name) and state is not None:
        return chess_state_meta_from_hashes(getattr(state, "_chess_history_hashes", None))
    if is_go_game(game_name) and state is not None:
        return {
            "go_ruleset": cfg.get("go_ruleset", "chinese"),
            "go_scoring": cfg.get("go_scoring", "area"),
            "go_komi": float(cfg.get("go_komi", 7.5)),
            "go_allow_suicide": bool(cfg.get("go_allow_suicide", False)),
            "passes": int(getattr(state, "_passes", 0)),
            "ko_point": int(getattr(state, "_ko_point", -1) if getattr(state, "_ko_point", None) is not None else -1),
            "black_caps": int(getattr(state, "_black_caps", 0)),
            "white_caps": int(getattr(state, "_white_caps", 0)),
        }
    return {}


def chess960_start_fen(index):
    if index < 0 or index >= 960:
        raise ValueError("Chess960 position index must be in [0, 959]")

    back = [0] * 8
    n = int(index)

    lb = (n % 4) * 2 + 1
    n //= 4
    back[lb] = 3

    db = (n % 4) * 2
    n //= 4
    back[db] = 3

    q_idx = n % 6
    n //= 6
    empty_idx = 0
    for i in range(8):
        if back[i] == 0:
            if empty_idx == q_idx:
                back[i] = 5
                break
            empty_idx += 1

    knight_table = [
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    ]
    kn1, kn2 = knight_table[n]
    empties = [i for i, piece in enumerate(back) if piece == 0]
    back[empties[kn1]] = 2
    back[empties[kn2]] = 2

    remaining = [i for i, piece in enumerate(back) if piece == 0]
    back[remaining[0]] = 4
    back[remaining[1]] = 6
    back[remaining[2]] = 4

    piece_map = {2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
    white_back = "".join(piece_map[piece] for piece in back)
    black_back = white_back.lower()
    rights = (
        chr(ord("A") + remaining[2]) +
        chr(ord("A") + remaining[0]) +
        chr(ord("a") + remaining[2]) +
        chr(ord("a") + remaining[0])
    )
    return f"{black_back}/pppppppp/8/8/8/8/PPPPPPPP/{white_back} w {rights} - 0 1"


def initial_chess_fen(cfg, rng=None):
    if not cfg.get("chess960", False):
        return STANDARD_CHESS_FEN
    index = cfg.get("chess960_index")
    if index is None:
        picker = rng if rng is not None else random
        index = picker.randrange(960)
    return chess960_start_fen(int(index))


def encode_chess_fen(fen):
    """Encode a chess FEN to 36-channel tensor (AlphaZero-complete for t=0).
    0-5: my pieces, 6-11: opp pieces (relative), 12-13: repetition,
    14-27: history (zero), 28: color, 29: move count,
    30-33: castling, 34: halfmove clock, 35: EP."""
    enc = np.zeros((36, 8, 8), dtype=np.float32)
    parts = fen.split()
    if len(parts) < 4:
        return enc

    board_part, side_part, castling_part, ep_part = parts[:4]
    is_white = (side_part != 'b')
    # Piece planes: relative to side-to-move
    white_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
    black_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    rank = 7
    file = 0
    white_king_file = 4
    for ch in board_part:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        elif ch in white_map:
            plane = white_map[ch] if is_white else (white_map[ch] + 6)
            enc[plane, rank, file] = 1.0
            if ch == 'K':
                white_king_file = file
            file += 1
        elif ch in black_map:
            plane = (black_map[ch] + 6) if is_white else black_map[ch]
            enc[plane, rank, file] = 1.0
            file += 1

    # Plane 28: side to move
    if is_white:
        enc[28] = 1.0

    # Plane 29: move count (from FEN fullmove field)
    if len(parts) >= 6:
        try:
            fullmove = int(parts[5])
            enc[29] = min(fullmove / 200.0, 1.0)
        except ValueError:
            pass

    # Planes 30-33: castling rights
    for ch in castling_part:
        if ch == 'K':
            enc[30] = 1.0
        elif ch == 'Q':
            enc[31] = 1.0
        elif ch == 'k':
            enc[32] = 1.0
        elif ch == 'q':
            enc[33] = 1.0
        elif 'A' <= ch <= 'H':
            rook_file = ord(ch) - ord('A')
            enc[30 if rook_file > white_king_file else 31] = 1.0
        elif 'a' <= ch <= 'h':
            rook_file = ord(ch) - ord('a')
            enc[32 if rook_file > white_king_file else 33] = 1.0

    # Plane 34: halfmove clock
    if len(parts) >= 5:
        try:
            half = int(parts[4])
            enc[34] = min(half / 100.0, 1.0)
        except ValueError:
            pass

    # Plane 35: en passant target
    if ep_part != "-" and len(ep_part) >= 2:
        ep_file = ord(ep_part[0]) - ord('a')
        ep_rank = ord(ep_part[1]) - ord('1')
        if 0 <= ep_file < 8 and 0 <= ep_rank < 8:
            enc[35, ep_rank, ep_file] = 1.0
    return enc

# ═══════════════════════════════════════════
# § NN Architecture
# ═══════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch//r), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        w = x.mean(dim=(2,3))
        return x * self.fc(w).unsqueeze(-1).unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(self, ch, se=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1,bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch,ch,3,padding=1,bias=False))
        self.se = SEBlock(ch) if se else None
    def forward(self, x):
        out = self.net(x)
        if self.se: out = self.se(out)
        return x + out

class AlphaZeroNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch, bs = cfg['filters'], cfg['board']
        n2 = bs * bs
        self.input_conv = nn.Sequential(nn.Conv2d(cfg['ch'],ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        blocks = [ResBlock(ch, se=(i >= cfg['blocks']-2)) for i in range(cfg['blocks'])]
        self.tower = nn.Sequential(*blocks)
        self.p_head = nn.Sequential(nn.Conv2d(ch,32,1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Conv2d(32,4,1,bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.p_fc = nn.Linear(4*n2, cfg['actions'])
        self.v_head = nn.Sequential(nn.Conv2d(ch,32,1,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(32, cfg['vh']), nn.ReLU(), nn.Linear(cfg['vh'], 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.input_conv(x))
        p = self.p_head(h).reshape(h.size(0),-1)
        p = self.p_fc(p)
        v = self.v_head(h).mean(dim=(2,3))
        v = self.v_fc(v).squeeze(-1)
        return p, v

# ═══════════════════════════════════════════
# § Replay Buffer
# ═══════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity, recent_fraction=0.0, recent_window=0):
        self.buf = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.recent_fraction = float(max(0.0, min(1.0, recent_fraction)))
        self.recent_window = int(max(0, recent_window))

    def add(self, state, policy, value):
        with self._lock:
            self.buf.append((state, policy, value))

    def add_game(self, states, policies, outcome, start_player=1):
        """Add full game using values from each position's side-to-move.

        `outcome` is encoded in absolute player colors: +1 first player wins,
        -1 second player wins, 0 draw. Targets stored in replay must instead be
        from the current side-to-move perspective at each position.
        """
        with self._lock:
            for i, (s, p) in enumerate(zip(states, policies)):
                player_to_move = start_player if (i % 2 == 0) else -start_player
                val = outcome * player_to_move
                self.buf.append((s, p, val))

    def _sample_indices_from_size(self, total_size, n):
        sample_n = min(n, total_size)
        if sample_n <= 0:
            return []
        if self.recent_fraction <= 0.0 or self.recent_window <= 0:
            return random.sample(range(total_size), sample_n)

        recent_start = max(0, total_size - min(self.recent_window, total_size))
        recent_indices = range(recent_start, total_size)
        older_indices = range(0, recent_start)

        recent_target = min(sample_n, math.ceil(sample_n * self.recent_fraction))
        n_recent = min(recent_target, total_size - recent_start)
        n_older = min(sample_n - n_recent, recent_start)

        chosen = []
        if n_recent > 0:
            chosen.extend(random.sample(recent_indices, n_recent))
        if n_older > 0:
            chosen.extend(random.sample(older_indices, n_older))

        if len(chosen) < sample_n:
            chosen_set = set(chosen)
            remaining_pool = [i for i in range(total_size) if i not in chosen_set]
            chosen.extend(random.sample(remaining_pool, sample_n - len(chosen)))
        return chosen

    def _sample_indices_locked(self, n):
        return self._sample_indices_from_size(len(self.buf), n)

    def _sample_examples_locked(self, n):
        indices = self._sample_indices_locked(n)
        return [self.buf[i] for i in indices]

    def _sample_examples_from_snapshot(self, snapshot, n):
        indices = self._sample_indices_from_size(len(snapshot), n)
        return [snapshot[i] for i in indices]

    def sample(self, n):
        with self._lock:
            batch = self._sample_examples_locked(n)
        return collate_replay_samples(batch)

    def build_dataloader(self, batch_size, n_steps, pin_memory=False):
        with self._lock:
            snapshot = tuple(self.buf)
        examples = []
        for _ in range(max(0, int(n_steps))):
            examples.extend(self._sample_examples_from_snapshot(snapshot, batch_size))
        if not examples:
            return None
        dataset = ReplayDataset(examples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_replay_samples,
            drop_last=False,
        )

    def __len__(self):
        return len(self.buf)

    def save(self, path):
        data = list(self.buf)
        np.savez_compressed(path,
            states=np.array([d[0] for d in data]),
            policies=np.array([d[1] for d in data]),
            values=np.array([d[2] for d in data], dtype=np.float32))

    def load(self, path):
        if not os.path.exists(path): return 0
        d = np.load(path)
        n = len(d['states'])
        for i in range(n):
            self.buf.append((d['states'][i], d['policies'][i], float(d['values'][i])))
        return n


class ReplayDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_replay_samples(batch):
    if not batch:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )
    states = torch.from_numpy(np.stack([np.asarray(b[0], dtype=np.float32) for b in batch]))
    policies = torch.from_numpy(np.stack([np.asarray(b[1], dtype=np.float32) for b in batch]))
    values = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return states, policies, values

# ═══════════════════════════════════════════
# § TreeMCTS (arena evaluation only — not used for training)
# ═══════════════════════════════════════════

class MCTSNode:
    """Array-based MCTS node — children stored as parallel numpy arrays.
    
    Optimization: vectorized PUCT selection via numpy instead of Python for-loop.
    Vectorized PUCT provides significant speedup over per-child-object iteration.
    """
    __slots__ = ['move', 'child_moves', 'child_n', 'child_w', 'child_prior',
                 'child_nodes', 'n_children', 'is_expanded', 'total_n']

    def __init__(self, parent=None, move=None, prior=0.0):
        self.move = move if move is not None else -1
        self.child_moves = None
        self.child_n = None
        self.child_w = None
        self.child_prior = None
        self.child_nodes = None  # dict: child_idx → MCTSNode
        self.n_children = 0
        self.is_expanded = False
        self.total_n = 0

    def expand(self, legal_moves, priors):
        k = len(legal_moves)
        self.child_moves = np.array(legal_moves, dtype=np.int32)
        self.child_n = np.zeros(k, dtype=np.int32)
        self.child_w = np.zeros(k, dtype=np.float32)
        p = np.array([priors[m] if m < len(priors) else 1.0/max(k,1)
                       for m in legal_moves], dtype=np.float32)
        ps = p.sum()
        if ps > 0: p /= ps
        self.child_prior = p
        self.child_nodes = {}
        self.n_children = k
        self.is_expanded = True
        self.total_n = 0

    def select_child_vectorized(self, c_puct):
        """Vectorized PUCT: one numpy op instead of per-child Python loop."""
        n = self.child_n
        unvisited = (n == 0)
        if unvisited.any():
            scores = np.where(unvisited,
                              1e6 + self.child_prior * 1000 + np.random.random(self.n_children) * 0.01,
                              -1e9)
            return int(np.argmax(scores))
        q = self.child_w / np.maximum(n.astype(np.float32), 1)
        sqrt_total = math.sqrt(self.total_n + 1)
        u = c_puct * self.child_prior * sqrt_total / (1 + n.astype(np.float32))
        return int(np.argmax(q + u))

    def backup_child(self, ci, value):
        self.child_n[ci] += 1
        self.child_w[ci] += value
        self.total_n += 1

    @property
    def n(self):
        return self.total_n

    @property
    def w(self):
        return float(self.child_w.sum()) if self.child_w is not None else 0.0

    @property
    def children(self):
        """Compatibility: iterate child nodes (for arena/eval code)."""
        if self.child_nodes is None: return []
        return list(self.child_nodes.values())


# ──── LEGACY: Arena evaluation helper only (not used in training) ────
class TreeMCTS:

    """Optimized tree MCTS with array-based nodes and vectorized PUCT.
    
    select → expand → evaluate → backup.
    Children stored as numpy arrays per node; PUCT computed in one numpy op.
    
    Accuracy enhancements (speed-neutral, −10% overhead):
    - Heuristic prior at root: threat/adjacency patterns for gomoku/go (1 call only)
    - Fast leaf value: O(4) line check from last move (not full-board scan)
    - FPU reduction: parent_value - offset for unvisited children
    
    Accuracy enhancements improve forced-move detection and H2H win rate vs uniform baseline.
    """
    FPU_OFFSET = 0.25
    FPU_PRIOR_WEIGHT = 3.0

    def __init__(self, cfg, model=None, device='cpu'):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.n_actions = cfg['actions']
        self.board_size = cfg['board']
        self.penalty_mode = cfg.get('penalty_mode', 'GatedRefresh')
        self.c_puct = cfg.get('c_puct', 2.0)
        self._win_len = cfg.get('win', 0)
        self._encoder = cfg.get('_encoder')
        self._has_heuristic = hasattr(self._encoder, 'heuristic_prior') if self._encoder else False

    def _gomoku_heuristic_prior(self, board, player):
        """Delegate to encoder's heuristic_prior (game-agnostic)."""
        if self._encoder is not None:
            return self._encoder.heuristic_prior(board, player)
        n2 = self.board_size ** 2
        legal = np.zeros(self.n_actions, dtype=np.float32)
        for i in range(min(n2, self.n_actions)):
            if board[i] == 0: legal[i] = 1.0
        s = legal.sum()
        return legal / s if s > 0 else np.ones(self.n_actions, dtype=np.float32) / self.n_actions

    def _fast_leaf_value(self, board, last_move, player_who_moved):
        """Delegate to encoder's fast_leaf_value (game-agnostic)."""
        if self._encoder is not None:
            return self._encoder.fast_leaf_value(board, last_move, player_who_moved)
        return 0.0

    def _apply_move_board(self, board, player, action):
        new_board = board.copy()
        new_board[action] = player
        won = False
        wl = self.cfg.get('win', 0)
        if wl > 0:
            bs = self.board_size
            r0, c0 = action // bs, action % bs
            for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
                cnt = 1
                for sign in (1,-1):
                    nr, nc = r0+sign*dr, c0+sign*dc
                    while 0<=nr<bs and 0<=nc<bs and new_board[nr*bs+nc]==player:
                        cnt += 1; nr += sign*dr; nc += sign*dc
                if cnt >= wl: won = True; break
        return new_board, -player, won

    def _encode(self, board, player):
        return encode_board(self.cfg, board, player)

    def _evaluate_leaf(self, board, player):
        if self.model is not None:
            enc = self._encode(board, player)
            with torch.no_grad():
                x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, val = self.model(x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return probs, val.item()
        else:
            probs = np.ones(self.n_actions, dtype=np.float32)
            n2 = self.board_size ** 2
            for i in range(min(n2, self.n_actions)):
                if board[i] != 0: probs[i] = 0
            s = probs.sum()
            if s > 0: probs /= s
            return probs, 0.0

    def search(self, board_enc, player, legal_mask, n_iters):
        """Run tree MCTS (used by arena evaluation).
        
        """
        board = decode_board(self.cfg, board_enc, player)

        bs = self.board_size
        legal_indices = [i for i in range(min(self.n_actions, bs*bs)) if board[i] == 0]
        if not legal_indices:
            return np.zeros(self.n_actions, dtype=np.float32)

        # Root prior
        if self.model is not None:
            priors, root_val = self._evaluate_leaf(board, player)
        elif self._has_heuristic:
            priors = self._gomoku_heuristic_prior(board, player)
            root_val = 0.0
        else:
            priors, root_val = self._evaluate_leaf(board, player)

        root = MCTSNode()
        root.expand(legal_indices, priors)

        state = _SearchState(root=root, board=board, player=player, root_val=root_val, cfg=self)

        for it in range(n_iters):
            leaf = state.select_to_leaf()
            if leaf.is_terminal:
                state.backup(leaf, leaf.terminal_value)
            elif self.model is not None:
                priors, val = self._evaluate_leaf(leaf.board, leaf.player)
                state.expand_and_backup(leaf, priors, val)
            else:
                child_uniform = np.ones(self.n_actions, dtype=np.float32) / max(self.n_actions, 1)
                val = self._fast_leaf_value(leaf.board, leaf.last_move, -leaf.player)
                state.expand_and_backup(leaf, child_uniform, val)

        return state.extract_policy(self.n_actions)

@dataclass
class _LeafInfo:
    """Information about a leaf node reached during selection."""
    node: MCTSNode
    ci: int                    # child index in parent
    path: list                 # [(node, ci), ...]
    board: np.ndarray          # board state at leaf
    player: int                # player to move at leaf
    last_move: int             # last move that led here
    is_terminal: bool = False  # game over?
    terminal_value: float = 0.0


class _SearchState:
    """Mutable search state for one MCTS tree. Supports split select/expand/backup."""
    
    def __init__(self, root, board, player, root_val, cfg):
        self.root = root
        self.board = board
        self.player = player
        self.root_val = root_val
        self.cfg = cfg
        self._wl = cfg._win_len
        self._bs = cfg.board_size
        self._c_puct = cfg.c_puct
        self._penalty = cfg.penalty_mode
        self._fpu_off = cfg.FPU_OFFSET
        self._fpu_pw = cfg.FPU_PRIOR_WEIGHT

    def select_to_leaf(self):
        """SELECT phase: traverse tree to an unexpanded leaf. Returns _LeafInfo."""
        node = self.root
        cur_board = self.board.copy()
        cur_player = self.player
        path = []
        parent_value = self.root_val
        last_move = -1
        bs = self._bs; wl = self._wl; c_puct = self._c_puct
        ci = -1

        while node.is_expanded and node.n_children > 0:
            cp = c_puct
            if node is self.root and self._penalty == "GatedRefresh":
                cp = c_puct * 0.85
            elif node is self.root and self._penalty == "SelfAdaptive":
                cp = c_puct * 0.80

            n = node.child_n; p = node.child_prior
            unvisited = (n == 0)
            if unvisited.any():
                fpu_scores = parent_value - self._fpu_off + p * self._fpu_pw
                scores = np.where(unvisited,
                                  fpu_scores + np.random.random(node.n_children) * 0.001,
                                  -1e9)
                ci = int(np.argmax(scores))
            else:
                q = node.child_w / np.maximum(n.astype(np.float32), 1)
                sqrt_total = math.sqrt(node.total_n + 1)
                u = cp * p * sqrt_total / (1 + n.astype(np.float32))
                ci = int(np.argmax(q + u))
                parent_value = float(q[ci])

            path.append((node, ci))
            move = int(node.child_moves[ci])
            cur_board[move] = cur_player
            last_move = move

            # Win check
            won = False
            if wl > 0:
                r0, c0 = move // bs, move % bs
                for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
                    cnt = 1
                    for sign in (1,-1):
                        nr, nc = r0+sign*dr, c0+sign*dc
                        while 0<=nr<bs and 0<=nc<bs and cur_board[nr*bs+nc]==cur_player:
                            cnt += 1; nr += sign*dr; nc += sign*dc
                    if cnt >= wl: won = True; break
                if won:
                    return _LeafInfo(node=node, ci=ci, path=path, board=cur_board,
                                    player=cur_player, last_move=last_move,
                                    is_terminal=True, terminal_value=-1.0)

            cur_player = -cur_player
            if ci in node.child_nodes:
                node = node.child_nodes[ci]
            else:
                break

        return _LeafInfo(node=node, ci=ci, path=path, board=cur_board,
                         player=cur_player, last_move=last_move)

    def expand_and_backup(self, leaf, priors, leaf_val):
        """EXPAND leaf with priors, then BACKUP value through path."""
        bs = self._bs; na = self.cfg.n_actions
        child_legal = [i for i in range(min(na, bs*bs)) if leaf.board[i] == 0]
        if child_legal:
            new_node = MCTSNode()
            new_node.expand(child_legal, priors)
            leaf.node.child_nodes[leaf.ci] = new_node
            value = -leaf_val
        else:
            value = 0.0
        self._do_backup(leaf.path, value)

    def backup(self, leaf, value):
        """BACKUP a terminal value through the path."""
        self._do_backup(leaf.path, value)

    def _do_backup(self, path, value):
        for nd, ci in reversed(path):
            nd.backup_child(ci, value)
            value = -value

    def extract_policy(self, n_actions):
        visits = np.zeros(n_actions, dtype=np.float32)
        for i in range(self.root.n_children):
            if self.root.child_moves[i] < n_actions:
                visits[self.root.child_moves[i]] = self.root.child_n[i]
        total = visits.sum()
        if total > 0: visits /= total
        return visits


# ═══════════════════════════════════════════
# § Rust Server Self-Play
# ═══════════════════════════════════════════

def selfplay_rust(cfg, n_games, rust_binary="./target/release/mcts_demo"):
    """Run self-play via persistent Rust MCTS server (single process, multiple requests)."""
    rust_game = rust_game_name(cfg['_name'])

    all_trajectories = []
    remaining = n_games

    try:
        # Start persistent Rust server process
        proc = launch_rust_server(rust_binary)
    except FileNotFoundError:
        print(f"  [WARN] Rust binary not found: {rust_binary}", file=sys.stderr)
        return [], n_games
    except RuntimeError as e:
        print(f"  [WARN] {e}", file=sys.stderr)
        return [], n_games

    with tqdm(total=n_games, desc="Self-play (Rust)", leave=False) as pbar:
        while remaining > 0:
            batch = min(remaining, 5)
            req_dict = {
                "cmd": "selfplay",
                "game": rust_game,
                "iters": cfg['iters'],
                "n_games": batch,
                "temp_threshold": cfg['temp_th'],
            }
            if is_go_game(cfg.get('_name', '')):
                req_dict.update({
                    "go_ruleset": cfg.get("go_ruleset", "chinese"),
                    "go_scoring": cfg.get("go_scoring", "area"),
                    "go_komi": float(cfg.get("go_komi", 7.5)),
                    "go_allow_suicide": bool(cfg.get("go_allow_suicide", False)),
                })
            elif cfg.get("chess960", False):
                if cfg.get("chess960_index") is not None:
                    req_dict["chess960_index"] = int(cfg["chess960_index"])
                else:
                    req_dict["chess960_random_start"] = True
            req_dict.update(rust_search_options(cfg))
            try:
                proc_write_json_line(proc, req_dict)
                line = proc_read_json_line(proc)
                if line:
                    games = json_loads_fast(line)
                    for g in games:
                        all_trajectories.append(g)
                        pbar.update(1)
                    remaining -= batch
                else:
                    print(f"  [WARN] Rust server returned empty, falling back", file=sys.stderr)
                    break
            except (json.JSONDecodeError, BrokenPipeError, OSError) as e:
                print(f"  [WARN] Rust server error ({e}), falling back", file=sys.stderr)
                break

    # Gracefully terminate server
    try:
        proc_write_json_line(proc, {"cmd": "quit"})
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    return all_trajectories, remaining


# ═══════════════════════════════════════════
# § NN-Backed Rust Search Client (search_nn protocol)
# ═══════════════════════════════════════════

class NNSearchClient:
    """Drives bidirectional NN eval protocol with Rust MCTS server.
    
    Protocol:
      Python → Rust: {"cmd":"search_nn", "board":[...], "player":1, "iters":200, ...}
      Rust → Python: QIPC binary eval frame(s) on the same stdio pipe
      Python → Rust: QIPC binary eval response frame(s)
      Rust → Python: {"result":{"best_move":42,"policy":[...],...}}
    
    This gives Rust's fast MCTS with Python's NN evaluation.
    """
    def __init__(self, model, cfg, device, rust_binary="./target/release/mcts_demo"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.proc = None
        self.rust_binary = rust_binary
        self.search_read_timeout_s = float(
            os.environ.get("QUARTZ_SEARCH_STALL_TIMEOUT_S", "120") or 120.0
        )

    def start(self):
        self.proc = launch_rust_server(self.rust_binary)

    def stop(self):
        if self.proc:
            try:
                proc_write_json_line(self.proc, {"cmd": "quit"})
                self.proc.wait(timeout=5)
            except Exception: self.proc.kill()
            _cleanup_qipc_transport(self.proc)
            self.proc = None

    def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
        """Send position to Rust, handle eval callbacks, get result."""
        game_name = rust_game_name(self.cfg['_name'])
        last_error = None
        for attempt in range(2):
            if not self.proc:
                self.start()
            req_dict = {
                "cmd": "search_nn",
                "game": game_name,
                "player": int(player),
                "iters": self.cfg['iters'],
            }
            req_dict.update(rust_search_options(self.cfg, penalty_mode=penalty_mode))
            if is_chess_game(game_name) and fen:
                req_dict["fen"] = fen
                if state_meta:
                    req_dict.update(state_meta)
            else:
                req_dict["board"] = normalize_rust_board(game_name, board_flat)
                if state_meta:
                    req_dict.update(state_meta)
            try:
                payload = self._exchange_search_request(req_dict)
                if isinstance(payload, dict) and "result" in payload:
                    return payload.get("result", {})
                return payload if isinstance(payload, dict) else {}
            except TimeoutError as exc:
                last_error = exc
                logging.getLogger(__name__).warning(
                    "search_move timed out on attempt %d/2 for %s; restarting Rust server",
                    attempt + 1,
                    game_name,
                )
                self.stop()
        if last_error is not None:
            raise last_error
        return {}

    def search_moves_multi(self, jobs, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        if not jobs:
            return []
        req_dict = {
            "cmd": "search_nn_multi",
            "game": rust_game_name(self.cfg['_name']),
            "iters": self.cfg["iters"],
            "jobs": jobs,
        }
        req_dict.update(rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload.get("results", [])
        return []

    def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "search_nn_multi_session_open",
            "game": rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "jobs": jobs,
        }
        req_dict.update(rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload
        return {}

    def step_search_session(self, session_id, updates):
        if not self.proc:
            self.start()
        payload = self._exchange_search_request({
            "cmd": "search_nn_multi_session_step",
            "session_id": int(session_id),
            "updates": updates,
        })
        if isinstance(payload, dict):
            return payload
        return {}

    def close_search_session(self, session_id):
        if not self.proc:
            return {}
        try:
            proc_write_json_line(self.proc, {
                "cmd": "search_nn_multi_session_close",
                "session_id": int(session_id),
            })
            payload = proc_read_json_line(self.proc)
        except Exception:
            return {}
        if not payload:
            return {}
        try:
            return json_loads_fast(payload)
        except Exception:
            return {}

    def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "eval_nn_run",
            "game": rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "sessions": sessions,
            "max_moves": int(max_moves),
        }
        req_dict.update(rust_search_options(self.cfg, penalty_mode=penalty_mode))
        payload = self._exchange_search_request(req_dict)
        if isinstance(payload, dict):
            return payload
        return {}

    def selfplay_run(self, n_games, parallel, temp_threshold, penalty_mode="GatedRefresh", seed=0,
                     on_chunk=None, on_progress=None):
        if not self.proc:
            self.start()
        req_dict = {
            "cmd": "selfplay_nn_run",
            "game": rust_game_name(self.cfg["_name"]),
            "iters": self.cfg["iters"],
            "n_games": int(n_games),
            "parallel": int(parallel),
            "temp_threshold": int(temp_threshold),
            "seed": int(seed),
        }
        req_dict.update(rust_search_options(self.cfg, penalty_mode=penalty_mode))
        proc_write_json_line(self.proc, req_dict)

        # --- SHM ring buffer fast path ---
        ring = getattr(self.proc, "_quartz_ring_buffer", None)
        if ring is not None:
            aggregated_games = []
            def _on_json(obj):
                if not isinstance(obj, dict):
                    return
                if "selfplay_chunk" in obj:
                    games = obj["selfplay_chunk"].get("games", []) or []
                    if callable(on_chunk):
                        on_chunk(games)
                    else:
                        aggregated_games.extend(games)
                elif "selfplay_progress" in obj:
                    if callable(on_progress):
                        on_progress(obj["selfplay_progress"])
            ring_payload = _shm_eval_loop(ring, self.model, self.device, self.cfg, self.proc, on_json=_on_json)
            if isinstance(ring_payload, dict):
                if "selfplay_done" in ring_payload:
                    done = dict(ring_payload["selfplay_done"])
                    done["games"] = aggregated_games
                    return done
                if "games" in ring_payload:
                    payload = dict(ring_payload)
                    payload.setdefault("games", aggregated_games)
                    return payload
            # Read final result from stdout
            kind, payload = proc_read_message(self.proc)
            if kind == "json" and isinstance(payload, dict):
                if "selfplay_done" in payload:
                    done = dict(payload["selfplay_done"])
                    done["games"] = aggregated_games
                    return done
                if "games" in payload:
                    return payload
                return payload
            return {"games": aggregated_games}

        def parse_eval_group(kind, payload):
            if kind is None:
                return None, None
            if kind == "frame":
                frame_kind, frame_payload = payload
                frame_kind, frame_payload = proc_decode_eval_frame(self.proc, frame_kind, frame_payload)
                if frame_kind == QIPC_BATCH_EVAL_REQ:
                    return _make_eval_request_group(
                        "binary_batch",
                        unpack_qipc_batch_eval_req(frame_payload),
                        gi=0,
                    ), None
                if frame_kind == QIPC_EVAL_REQ:
                    return _make_eval_request_group(
                        "binary_single",
                        [unpack_qipc_eval_req(frame_payload)],
                        gi=0,
                    ), None
                return None, {"error": f"unexpected IPC frame kind: {frame_kind}"}
            if kind == "json":
                if not payload:
                    return None, None
                if "batch_eval_req" in payload:
                    requests = payload["batch_eval_req"].get("requests", [])
                    return _make_eval_request_group(
                        "json_batch",
                        [
                            (int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))
                            for er in requests
                        ],
                        gi=0,
                    ), None
                if "eval_req" in payload:
                    er = payload["eval_req"]
                    return _make_eval_request_group(
                        "json_single",
                        [(int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))],
                        gi=0,
                    ), None
                return None, payload
            return None, {"error": "unexpected message"}

        search_opts = rust_search_options(self.cfg)
        base_collect_timeout_s = min(
            0.006,
            max(0.00075, float(search_opts.get("batch_timeout_us", 1500)) / 1_000_000.0 * 0.9),
        )
        base_target_eval_items = max(1, int(search_opts.get("batch_size", self.cfg.get("batch_size", 8))))
        batch_items_ema = float(base_target_eval_items)
        collect_wait_ema_s = 0.0
        deferred = None
        aggregated_games = []

        while True:
            kind, payload = deferred if deferred is not None else proc_read_message(self.proc)
            deferred = None
            if kind is None:
                return {"games": aggregated_games}
            first_group, terminal = parse_eval_group(kind, payload)
            if terminal is not None:
                if isinstance(terminal, dict):
                    if "selfplay_chunk" in terminal:
                        games = terminal["selfplay_chunk"].get("games", []) or []
                        if callable(on_chunk):
                            on_chunk(games)
                        else:
                            aggregated_games.extend(games)
                        continue
                    if "selfplay_progress" in terminal:
                        if callable(on_progress):
                            on_progress(terminal["selfplay_progress"])
                        continue
                    if "selfplay_done" in terminal:
                        done = dict(terminal["selfplay_done"])
                        done["games"] = aggregated_games
                        return done
                    if "games" in terminal:
                        return terminal
                return terminal

            eval_groups = [first_group]
            eval_item_count = len(first_group["requests"])
            dynamic_target_eval_items, dynamic_collect_timeout_s = compute_eval_collect_policy(
                base_target_eval_items,
                base_collect_timeout_s,
                batch_items_ema=batch_items_ema,
                wait_ema_s=collect_wait_ema_s,
            )
            collect_t0 = time.perf_counter()
            deadline = time.perf_counter() + dynamic_collect_timeout_s
            while eval_item_count < dynamic_target_eval_items:
                timeout_s = max(0.0, deadline - time.perf_counter())
                if timeout_s <= 0.0 or not wait_readable(self.proc.stdout, timeout_s):
                    break
                next_kind, next_payload = proc_read_message(self.proc)
                next_group, next_terminal = parse_eval_group(next_kind, next_payload)
                if next_terminal is not None:
                    deferred = (next_kind, next_payload)
                    break
                eval_groups.append(next_group)
                eval_item_count += len(next_group["requests"])

            merged_items = sum(len(group["requests"]) for group in eval_groups)
            collect_wait_s = time.perf_counter() - collect_t0
            batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
            collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)
            responses = _run_batched_eval_groups(eval_groups, self.model, self.device, self.cfg)
            for response_group in responses:
                _write_batched_eval_group(self.proc, response_group)

    @staticmethod
    def _emit_duty_cycle(duty):
        """Emit eval-loop duty-cycle timing for monitoring.
        Only active when QUARTZ_DUTY_CYCLE=1 is set."""
        if not os.environ.get("QUARTZ_DUTY_CYCLE"):
            return
        total = duty["read_s"] + duty["collect_s"] + duty["model_s"] + duty["write_s"]
        if total < 1e-9:
            return
        import sys as _sys
        msg = (
            f'  [DutyCycle] cycles={duty["cycles"]}'
            f' read={duty["read_s"]:.3f}s({duty["read_s"]/total*100:.0f}%)'
            f' collect={duty["collect_s"]:.3f}s({duty["collect_s"]/total*100:.0f}%)'
            f' model={duty["model_s"]:.3f}s({duty["model_s"]/total*100:.0f}%)'
            f' write={duty["write_s"]:.3f}s({duty["write_s"]/total*100:.0f}%)'
            f' total={total:.3f}s\n'
        )
        try:
            _sys.stderr.write(msg)
            _sys.stderr.flush()
        except Exception:
            pass

    def _exchange_search_request(self, req_dict):
        if not self.proc:
            self.start()
        read_timeout_s = max(1.0, float(self.search_read_timeout_s))

        def parse_eval_group(kind, payload):
            if kind is None:
                return None, None
            if kind == "frame":
                frame_kind, frame_payload = payload
                frame_kind, frame_payload = proc_decode_eval_frame(self.proc, frame_kind, frame_payload)
                if frame_kind == QIPC_BATCH_EVAL_REQ:
                    return _make_eval_request_group(
                        "binary_batch",
                        unpack_qipc_batch_eval_req(frame_payload),
                        gi=0,
                    ), None
                if frame_kind == QIPC_EVAL_REQ:
                    return _make_eval_request_group(
                        "binary_single",
                        [unpack_qipc_eval_req(frame_payload)],
                        gi=0,
                    ), None
                return None, {"error": f"unexpected IPC frame kind: {frame_kind}"}
            if kind == "json":
                if not payload:
                    return None, None
                if "batch_eval_req" in payload:
                    requests = payload["batch_eval_req"].get("requests", [])
                    return _make_eval_request_group(
                        "json_batch",
                        [
                            (int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))
                            for er in requests
                        ],
                        gi=0,
                    ), None
                if "eval_req" in payload:
                    er = payload["eval_req"]
                    return _make_eval_request_group(
                        "json_single",
                        [(int(er.get("num_actions", self.cfg["actions"])), er.get("features", []))],
                        gi=0,
                    ), None
                return None, payload
            return None, {"error": "unexpected message"}

        search_opts = rust_search_options(self.cfg)
        base_collect_timeout_s = min(
            0.006,
            max(0.00075, float(search_opts.get("batch_timeout_us", 1500)) / 1_000_000.0 * 0.9),
        )
        base_target_eval_items = max(1, int(search_opts.get("batch_size", self.cfg.get("batch_size", 8))))
        batch_items_ema = float(base_target_eval_items)
        collect_wait_ema_s = 0.0
        proc_write_json_line(self.proc, req_dict)

        # --- SHM ring buffer fast path ---
        ring = getattr(self.proc, "_quartz_ring_buffer", None)
        if ring is not None:
            ring_payload = _shm_eval_loop(ring, self.model, self.device, self.cfg, self.proc)
            if isinstance(ring_payload, dict):
                return ring_payload
            kind, payload = proc_read_message(self.proc, timeout_s=read_timeout_s)
            if kind == "json" and isinstance(payload, dict):
                return payload
            return {}

        deferred = None
        _duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
        _duty_log_interval = 16
        _use_pl = (
            not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
            and self.model is not None
            and not hasattr(self.model, "predict")
        )
        pipeline = None
        inflight = False
        pending_response = None
        if _use_pl:
            pipeline = InferencePipelineThread(self.model, self.device, self.cfg, max_pending=1)
            pipeline.start()

        try:
          while True:
            # --- Flush: if inflight, wait for inference and write result ---
            # This MUST happen before read, because Rust broker won't send
            # the next request until it receives the response for the current one.
            if inflight and pipeline is not None:
                model_t0 = time.perf_counter()
                responses = pipeline.collect(timeout=30.0)
                _duty["model_s"] += time.perf_counter() - model_t0
                inflight = False
                write_t0 = time.perf_counter()
                for rg in responses:
                    _write_batched_eval_group(self.proc, rg)
                _duty["write_s"] += time.perf_counter() - write_t0

            # --- Read first request ---
            read_t0 = time.perf_counter()
            kind, payload = deferred if deferred is not None else proc_read_message(self.proc, timeout_s=read_timeout_s)
            deferred = None
            _duty["read_s"] += time.perf_counter() - read_t0

            if kind is None:
                if _duty["cycles"] > 0:
                    NNSearchClient._emit_duty_cycle(_duty)
                return {}
            first_group, terminal = parse_eval_group(kind, payload)
            if terminal is not None:
                if _duty["cycles"] > 0:
                    NNSearchClient._emit_duty_cycle(_duty)
                return terminal

            # --- Collect batch ---
            eval_groups = [first_group]
            eval_item_count = len(first_group["requests"])
            dynamic_target_eval_items, dynamic_collect_timeout_s = compute_eval_collect_policy(
                base_target_eval_items,
                base_collect_timeout_s,
                batch_items_ema=batch_items_ema,
                wait_ema_s=collect_wait_ema_s,
            )
            collect_t0 = time.perf_counter()
            deadline = time.perf_counter() + dynamic_collect_timeout_s
            while eval_item_count < dynamic_target_eval_items:
                timeout_s = max(0.0, deadline - time.perf_counter())
                if timeout_s <= 0.0 or not wait_readable(self.proc.stdout, timeout_s):
                    break
                next_kind, next_payload = proc_read_message(self.proc, timeout_s=read_timeout_s)
                next_group, next_terminal = parse_eval_group(next_kind, next_payload)
                if next_terminal is not None:
                    deferred = (next_kind, next_payload)
                    break
                eval_groups.append(next_group)
                eval_item_count += len(next_group["requests"])

            merged_items = sum(len(group["requests"]) for group in eval_groups)
            collect_wait_s = time.perf_counter() - collect_t0
            _duty["collect_s"] += collect_wait_s
            batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
            collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)

            # --- Inference: submit to pipeline or run sync ---
            if pipeline is not None:
                pipeline.submit(eval_groups)
                inflight = True
                # Pipeline overlap: inference runs in background while we loop
                # back to flush (collect result + write) then read next batch.
                # The overlap happens when Rust has already queued the next
                # batch in the broker channel before we finish writing.
            else:
                model_t0 = time.perf_counter()
                responses = _run_batched_eval_groups(eval_groups, self.model, self.device, self.cfg)
                _duty["model_s"] += time.perf_counter() - model_t0
                write_t0 = time.perf_counter()
                for rg in responses:
                    _write_batched_eval_group(self.proc, rg)
                _duty["write_s"] += time.perf_counter() - write_t0

            _duty["cycles"] += 1
            if _duty["cycles"] % _duty_log_interval == 0:
                NNSearchClient._emit_duty_cycle(_duty)
        finally:
            if inflight and pipeline is not None:
                try:
                    drain = pipeline.collect(timeout=10.0)
                    for rg in drain:
                        _write_batched_eval_group(self.proc, rg)
                except Exception:
                    pass
            if pipeline is not None:
                pipeline.stop()

    def _eval_from_features(self, features, n_act):
        try:
            ch, bs = self.cfg['ch'], self.cfg['board']
            expected = ch * bs * bs

            if len(features) == expected and self.model is not None:
                x = np.asarray(features, dtype=np.float32).reshape(1, ch, bs, bs)
                with torch.inference_mode():
                    probs, vals_np = _run_model_batch(self.model, self.device, x)
                return probs[0][:n_act], float(vals_np[0])
        except Exception as e:
            import sys; print(f'[WARN] NN eval failed: {e}', file=sys.stderr)
        # Fallback uniform
        na = max(1, int(n_act))
        return np.full(na, 1.0 / na, dtype=np.float32), 0.0

    def _eval_nn_json(self, eval_req):
        """Legacy JSON eval protocol fallback."""
        policy, value = self._eval_from_features(
            eval_req.get("features", []),
            eval_req.get("num_actions", self.cfg['actions']))
        return {"eval_resp": {"policy": policy.tolist(), "value": value}}

    def _eval_nn_binary(self, payload):
        num_actions, features, _model_tag = unpack_qipc_eval_req(payload)
        policy, value = self._eval_from_features(features, num_actions)
        return pack_qipc_eval_resp(policy, value)

    def _eval_nn_batch_json(self, batch_req):
        """Legacy JSON batch eval protocol fallback.

        Protocol:
          Rust → Python: {"batch_eval_req":{"batch_size":N,"requests":[{"features":[...],"num_actions":225},...]}}
          Python → Rust: {"batch_eval_resp":{"responses":[{"policy":[...],"value":0.42},...]}}
        """
        try:
            requests = batch_req["requests"]
            batch_size = len(requests)
            ch, bs = self.cfg['ch'], self.cfg['board']
            expected = ch * bs * bs

            if self.model is not None and batch_size > 0:
                features_list = []
                for req in requests:
                    feats = req.get("features", [])
                    if len(feats) == expected:
                        features_list.append(feats)
                    else:
                        features_list.append([0.0] * expected)

                probs, vals_np = _run_model_batch(
                    self.model, self.device,
                    np.asarray(features_list, dtype=np.float32).reshape(batch_size, ch, bs, bs))

                responses = []
                for i, req in enumerate(requests):
                    na = req.get("num_actions", self.cfg['actions'])
                    responses.append({
                        "policy": probs[i][:na].tolist(),
                        "value": float(vals_np[i])
                    })
                return {"batch_eval_resp": {"responses": responses}}
        except Exception as e:
            import sys; print(f'[WARN] Batch NN eval failed: {e}', file=sys.stderr)
        # Fallback: uniform for all requests
        na = self.cfg['actions']
        uniform = {"policy": [1.0/max(1,na)]*na, "value": 0.0}
        n = batch_req.get("batch_size", 1) if 'batch_req' in dir() else 1
        return {"batch_eval_resp": {"responses": [uniform]*n}}

    def _eval_nn_batch_binary(self, payload):
        requests = unpack_qipc_batch_eval_req(payload)
        batch_size = len(requests)
        ch, bs = self.cfg['ch'], self.cfg['board']
        expected = ch * bs * bs
        if self.model is not None and batch_size > 0:
            features_list = []
            num_actions = []
            for na, feats, _model_tag in requests:
                num_actions.append(int(na))
                if feats.size == expected:
                    features_list.append(np.asarray(feats, dtype=np.float32).reshape(ch, bs, bs))
                else:
                    features_list.append(np.zeros((ch, bs, bs), dtype=np.float32))
            probs, vals_np = _run_model_batch(self.model, self.device, np.stack(features_list, axis=0))
            policies = [probs[i][:num_actions[i]] for i in range(batch_size)]
            values = [float(vals_np[i]) for i in range(batch_size)]
            return pack_qipc_batch_eval_resp(policies, values)
        uniform_policies = []
        for na, _ in requests:
            na = max(1, int(na))
            uniform_policies.append(np.full(na, 1.0 / na, dtype=np.float32))
        return pack_qipc_batch_eval_resp(uniform_policies, [0.0] * len(uniform_policies))


# ═══════════════════════════════════════════
# § Actor/Learner Separation: Background Self-Play Worker
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# § Rust NN-Backed Self-Play
# ═══════════════════════════════════════════

def selfplay_rust_nn(cfg, model, device, n_games, rust_binary="./target/release/mcts_demo"):
    """Self-play using Rust MCTS engine with Python NN evaluation.
    
    This is the GOLD STANDARD self-play path:
    - Rust does tree search (PUCT + QUARTZ controller + TT + virtual loss)
    - Python provides NN policy+value via bidirectional search_nn protocol
    - Full Rust search semantics, not the simplified Python TreeMCTS
    
    Flow per move:
      Python → Rust: {"cmd":"search_nn", "board":[...], "player":1, ...}
      Rust ↔ Python: multiple eval_req/eval_resp exchanges during MCTS
      Rust → Python: {"result":{"best_move":42, "policy":[...], ...}}
    """
    n_actions = cfg['actions']
    penalty_mode = cfg.get('penalty_mode', 'GatedRefresh')
    is_chess = is_chess_game(cfg.get('_name'))
    max_moves = 500 if is_chess or is_go_game(cfg.get('_name')) else (cfg['board'] ** 2 + 5)

    client = NNSearchClient(model, cfg, device, rust_binary)
    try:
        client.start()
    except FileNotFoundError:
        print(f"  [WARN] Rust binary not found: {rust_binary}", file=sys.stderr)
        return [], [], [], n_games
    except RuntimeError as e:
        print(f"  [WARN] {e}", file=sys.stderr)
        return [], [], [], n_games

    all_states, all_policies, all_outcomes = [], [], []

    with tqdm(total=n_games, desc="Self-play (Rust+NN)", leave=False) as pbar:
        for game_idx in range(n_games):
            game_states, game_policies = [], []
            move_count = 0
            outcome = 0.0
            void_result = False

            if is_chess:
                current_fen = initial_chess_fen(cfg)
                current_chess_meta = {}
                player = 1
                chess_outcome = 0.0  # updated on terminal detection
            else:
                game = build_training_game_adapter(cfg)

            while move_count < max_moves:
                # Encode state for training data
                if is_chess:
                    enc = encode_chess_fen(current_fen)
                else:
                    legal = game.legal_moves()
                    if not legal:
                        if hasattr(game, "is_void_result") and game.is_void_result():
                            void_result = True
                            outcome = 0.0
                        else:
                            outcome = float(game.outcome_for_black() or 0.0)
                        break
                    player = 1 if game.current_player() == 0 else -1
                    enc = game._encode()

                # Call Rust MCTS with NN evaluation
                if is_chess:
                    result = client.search_move(
                        None, player, penalty_mode, fen=current_fen, state_meta=current_chess_meta)
                else:
                    result = client.search_move(
                        game._board, player, penalty_mode,
                        state_meta=build_rust_state_meta(cfg.get('_name'), game, cfg))
                if not result or 'error' in result:
                    break

                # Extract policy
                pol_entries = result.get('policy', [])
                if not pol_entries:
                    # Terminal: Rust adjudicates mate, stalemate, dead positions,
                    # and automatic draw rules directly from the root FEN.
                    if is_chess:
                        terminal_value = result.get('value', 0.0)
                        # terminal_value is from current player's perspective
                        # Convert to first player (white=1) perspective
                        chess_outcome = terminal_value * player  # player=1(white)/-1(black)
                    break
                policy = dense_policy_from_sparse(pol_entries, n_actions)

                game_states.append(enc.copy())
                game_policies.append(policy.copy())

                if is_chess:
                    # Chess: Rust returns result_fen after applying best_move
                    new_fen = result.get('result_fen', '')
                    if not new_fen or new_fen == current_fen:
                        break  # game over
                    current_fen = new_fen
                    current_chess_meta = chess_state_meta_from_hashes(
                        result.get("result_history_hashes", []))
                    move_count += 1
                    player = -player
                else:
                    chosen = choose_selfplay_move(
                        policy, legal, move_count, cfg['temp_th'],
                        fallback_best=result.get('best_move', -1))
                    game.apply_move(chosen)
                    move_count += 1
                    if game.is_terminal():
                        if hasattr(game, "is_void_result") and game.is_void_result():
                            void_result = True
                            outcome = 0.0
                        else:
                            outcome = float(game.outcome_for_black() or 0.0)
                        break

            # Record game
            if is_chess:
                outcome = chess_outcome
            elif void_result:
                game_states = []
                game_policies = []
            all_states.append(game_states)
            all_policies.append(game_policies)
            all_outcomes.append(outcome)
            pbar.update(1)
            pbar.set_postfix_str(f"moves={move_count}")

    client.stop()
    return all_states, all_policies, all_outcomes, 0


def _estimate_selfplay_positions_per_game(cfg, recent_chunks):
    rolling_games = sum(int(chunk.get("games", 0) or 0) for chunk in recent_chunks)
    rolling_positions = sum(int(chunk.get("positions", 0) or 0) for chunk in recent_chunks)
    if rolling_games > 0 and rolling_positions > 0:
        return max(1.0, rolling_positions / rolling_games)
    board = int(cfg.get("board", 7) or 7)
    # Use board area as a game-length prior when no telemetry exists yet.
    return max(4.0, float(board * board) * 0.5)


def initial_replay_fill_target(cfg, recent_chunks):
    train_batch = max(1, int(cfg.get("batch", 256) or 256))
    batch_target = max(1, int(cfg.get("batch_size", 8) or 8))
    base_parallel = max(1, int(cfg.get("bg_parallel", 2) or 2))
    positions_per_game = _estimate_selfplay_positions_per_game(cfg, recent_chunks)
    warm_games = max(base_parallel, int(math.ceil(batch_target / max(positions_per_game, 1.0))))
    return int(min(train_batch, max(batch_target, int(math.ceil(warm_games * positions_per_game)))))


def plan_selfplay_runner_chunk(cfg, replay_size, recent_chunks):
    base_parallel = max(1, int(cfg.get("bg_parallel", 2) or 2))
    base_batch_games = max(1, int(cfg.get("bg_batch_games", base_parallel) or base_parallel))
    batch_target = max(1, int(cfg.get("batch_size", 8) or 8))
    train_batch = max(1, int(cfg.get("batch", 256) or 256))
    logical_threads = max(1, int(os.cpu_count() or base_parallel))
    positions_per_game = _estimate_selfplay_positions_per_game(cfg, recent_chunks)
    replay_deficit = max(0, train_batch - int(replay_size))
    games_needed = max(1, int(math.ceil(replay_deficit / max(positions_per_game, 1.0))))

    parallel = base_parallel
    batch_games = base_batch_games
    if cfg.get("_selfplay_runner_mode") == "rust_selfplay_state_machine":
        # Keep enough active slots to feed the NN batch target, and scale further
        # when the replay deficit exceeds recent per-game yield.
        parallel_cap = max(base_parallel, logical_threads)
        parallel = max(base_parallel, min(parallel_cap, max(batch_target, games_needed)))
        max_batch_games_cap = max(base_batch_games, min(train_batch, max(parallel * 4, batch_target * 4)))
        batch_games = min(max_batch_games_cap, max(base_batch_games, parallel, games_needed))

    return {
        "parallel": int(parallel),
        "batch_games": int(batch_games),
        "games_per_call": int(max(parallel, min(batch_games, max(base_parallel, batch_target)))),
        "replay_deficit": int(replay_deficit),
        "estimated_positions_per_game": round(float(positions_per_game), 3),
    }


class SelfPlayWorker:
    """Background actor: Rust MCTS + batched NN eval (Tier 2).
    
    Uses frozen model snapshot for NN evaluation via search_nn IPC.
    Produces full Tier 2 training data (TT+VL+PW+QUARTZ search quality).
    
    Backpressure: pauses when replay buffer is >80% full.
    """
    BACKPRESSURE_RATIO = 0.8
    BACKPRESSURE_SLEEP = 0.5
    REPLAY_STALL_TIMEOUT_S = 45.0
    def __init__(self, cfg, model, device, replay, rust_binary):
        self.cfg = cfg; self.device = device
        self.replay = replay; self.rust_binary = rust_binary
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread = None
        self.games_generated = 0
        self.positions_generated = 0
        self._prev_count = 0
        self._cycles = 0
        self._total_time = 0.0
        self._backpressure_waits = 0
        self._last_cycle_s = 0.0
        self._last_cycle_positions = 0
        self._last_cycle_games = 0
        self._recent_chunks = deque(maxlen=8)
        self._proc_pool = RustServerPool(self.rust_binary)
        self._model = clone_actor_model(model)
        self._active_proc = None  # tracks current in-flight Rust process for kill-on-pause
        self._last_progress_ts = time.time()
        self._last_error = None
        self._consecutive_errors = 0
        self._last_plan = None

    def update_model(self, model):
        """Refresh frozen model snapshot from training model."""
        self._model = clone_actor_model(model)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._pause.clear()
        if self._thread:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                logging.getLogger(__name__).warning(
                    "SelfPlayWorker did not stop within timeout")
        self._proc_pool.close()

    def pause(self, wait=True):
        self._pause.set()
        if not wait:
            return True
        # Quick check: already idle?
        if self._idle.wait(timeout=2.0):
            return True
        # Cooperative cancel: signal Rust to stop at next wave boundary
        proc = self._active_proc
        if proc is not None:
            ring = getattr(proc, "_quartz_ring_buffer", None)
            if ring is not None:
                ring.request_cancel()
        # Wait indefinitely for the current wave to finish
        self._idle.wait()
        return True

    def resume(self):
        self._consecutive_errors = 0  # reset after pause/kill cycle
        self._pause.clear()

    def telemetry(self):
        rolling_positions = sum(chunk["positions"] for chunk in self._recent_chunks)
        rolling_games = sum(chunk["games"] for chunk in self._recent_chunks)
        rolling_time = sum(chunk["elapsed_s"] for chunk in self._recent_chunks)
        rolling_positions_per_s = rolling_positions / max(rolling_time, 1e-6)
        mean_chunk_positions = (
            rolling_positions / max(len(self._recent_chunks), 1)
            if self._recent_chunks else 0.0
        )
        peak_chunk_positions = max(
            (chunk["positions"] for chunk in self._recent_chunks),
            default=0,
        )
        burst_ratio = (
            peak_chunk_positions / max(mean_chunk_positions, 1.0)
            if mean_chunk_positions > 0.0 else 1.0
        )
        return {
            "games": self.games_generated,
            "positions": self.positions_generated,
            "cycles": self._cycles,
            "avg_cycle_s": round(self._total_time / max(self._cycles, 1), 3),
            "last_cycle_s": round(self._last_cycle_s, 3),
            "last_cycle_positions": self._last_cycle_positions,
            "last_cycle_games": self._last_cycle_games,
            "rolling_cycle_s": round(rolling_time / max(len(self._recent_chunks), 1), 3),
            "rolling_positions_per_s": round(rolling_positions_per_s, 3),
            "rolling_positions": int(rolling_positions),
            "rolling_games": int(rolling_games),
            "burst_ratio": round(burst_ratio, 3),
            "backpressure_waits": self._backpressure_waits,
            "worker_alive": bool(self._thread.is_alive()) if self._thread is not None else False,
            "paused": bool(self._pause.is_set()),
            "idle": bool(self._idle.is_set()),
            "last_progress_age_s": round(max(0.0, time.time() - self._last_progress_ts), 3),
            "last_error": self._last_error,
            "consecutive_errors": self._consecutive_errors,
            "last_plan": self._last_plan,
            "path": "rust+nn",
        }

    def status(self):
        return {
            "alive": bool(self._thread.is_alive()) if self._thread is not None else False,
            "paused": bool(self._pause.is_set()),
            "idle": bool(self._idle.is_set()),
            "last_progress_age_s": max(0.0, time.time() - self._last_progress_ts),
            "last_error": self._last_error,
            "consecutive_errors": self._consecutive_errors,
        }

    def _run(self):
        while not self._stop.is_set():
            try:
                if self._pause.is_set():
                    self._idle.set()
                    time.sleep(0.1)
                    continue
                if hasattr(self.replay.buf, 'maxlen') and self.replay.buf.maxlen:
                    fill = len(self.replay) / self.replay.buf.maxlen
                    if fill > self.BACKPRESSURE_RATIO:
                        self._backpressure_waits += 1
                        self._idle.set()
                        time.sleep(self.BACKPRESSURE_SLEEP)
                        continue

                t0 = time.time()
                n_new = 0
                plan = plan_selfplay_runner_chunk(self.cfg, len(self.replay), self._recent_chunks)
                self._last_plan = dict(plan)
                batch_games = int(plan["batch_games"])
                parallel = int(plan["parallel"])
                remaining = batch_games
                # Stream replay in smaller chunks so learner does not idle until a
                # long 16-game batch completes.
                while remaining > 0 and not self._stop.is_set():
                    if self._pause.is_set():
                        break
                    chunk_t0 = time.time()
                    chunk_games = min(remaining, int(plan.get("games_per_call", parallel)))
                    streamed_positions = 0
                    streamed_games = 0

                    def _on_game_stream(gs, gp, out, _traces):
                        nonlocal n_new, streamed_positions, streamed_games
                        self.replay.add_game(gs, gp, out)
                        n_new += len(gs)
                        streamed_positions += len(gs)
                        streamed_games += 1
                        self._last_progress_ts = time.time()
                        self._last_error = None
                        self._consecutive_errors = 0

                    self._idle.clear()
                    try:
                        states, policies, outcomes, _ = selfplay_rust_nn_batched(
                            self.cfg, self._model, self.device, chunk_games,
                            self.rust_binary,
                            parallel=min(parallel, chunk_games),
                            show_progress=False,
                            proc_pool=self._proc_pool,
                            on_game=_on_game_stream if self.cfg.get("_selfplay_runner_mode") == "rust_selfplay_state_machine" else None,
                            active_proc_ref=self)
                    finally:
                        self._active_proc = None
                        self._idle.set()
                    chunk_positions = streamed_positions
                    if self.cfg.get("_selfplay_runner_mode") != "rust_selfplay_state_machine":
                        for gs, gp, out in zip(states, policies, outcomes):
                            self.replay.add_game(gs, gp, out)
                            n_new += len(gs)
                            chunk_positions += len(gs)
                    if chunk_positions > 0:
                        self._last_progress_ts = time.time()
                        self._last_error = None
                        self._consecutive_errors = 0
                    chunk_elapsed = max(time.time() - chunk_t0, 1e-6)
                    self.games_generated += len(states)
                    self.positions_generated += chunk_positions
                    self._recent_chunks.append({
                        "games": int(len(states)),
                        "positions": int(chunk_positions),
                        "elapsed_s": float(chunk_elapsed),
                    })
                    remaining -= chunk_games
                self._cycles += 1
                cycle_s = time.time() - t0
                self._total_time += cycle_s
                self._last_cycle_s = cycle_s
                self._last_cycle_positions = n_new
                self._last_cycle_games = batch_games
            except Exception as e:
                self._idle.set()  # ensure idle is set on any error
                self._active_proc = None
                self._last_error = str(e)
                self._consecutive_errors += 1
                if self._consecutive_errors <= 3:
                    logging.getLogger(__name__).exception(
                        "SelfPlayWorker error (%s): %r",
                        type(e).__name__,
                        e,
                    )
                time.sleep(min(self._consecutive_errors, 5))


def _encode_board_with_history(cfg, board_12_sequence, move_idx, player):
    """Encode board with AlphaZero-style 8-step history (17 channels).

    Args:
        cfg: game config
        board_12_sequence: list of all board snapshots (0/1/2 encoding) up to this point
        move_idx: index into board_12_sequence for the current position
        player: current player (+1 or -1)
    """
    bs = cfg['board']
    n2 = bs * bs
    history_len = 8
    total_ch = history_len * 2 + 1  # 17
    enc = np.zeros((total_ch, bs, bs), dtype=np.float32)

    for t in range(history_len):
        hist_idx = move_idx - t
        if hist_idx < 0:
            break  # no more history available
        board_12 = board_12_sequence[hist_idx]
        board_flat = np.asarray([
            1 if int(v) == 1 else (-1 if int(v) == 2 else 0)
            for v in board_12
        ], dtype=np.int8)
        plane_my = t * 2
        plane_opp = t * 2 + 1
        for i in range(min(n2, len(board_flat))):
            r, c = i // bs, i % bs
            if board_flat[i] == player:
                enc[plane_my, r, c] = 1.0
            elif board_flat[i] != 0:
                enc[plane_opp, r, c] = 1.0

    # Color plane (last channel)
    if player == 1:
        enc[total_ch - 1] = 1.0
    return enc


def _decode_streamed_selfplay_game(cfg, game_payload):
    n_actions = int(cfg["actions"])
    board_hist = game_payload.get("states", []) or []
    player_hist = game_payload.get("players", []) or []
    policy_hist = game_payload.get("policies", []) or []
    traces = game_payload.get("trace", []) or []
    states = []
    policies = []
    for move_idx, (board_12, raw_player, sparse_pol) in enumerate(
            zip(board_hist, player_hist, policy_hist)):
        player = 1 if int(raw_player) > 0 else -1
        # Encode with 8-step history from the sequence of board snapshots
        states.append(_encode_board_with_history(cfg, board_hist, move_idx, player))
        policies.append(dense_policy_from_sparse(sparse_pol, n_actions))
    outcome = float(game_payload.get("outcome", 0.0) or 0.0)
    return states, policies, outcome, traces


# ═══════════════════════════════════════════
# § Replay Metrics: Diversity + Freshness
# ═══════════════════════════════════════════

class ReplayMetrics:
    """Track replay buffer health metrics (Doc 22 P2).
    
    - freshness: fraction of buffer replaced this iteration
    - policy_entropy: average entropy of policy targets (diversity proxy)
    """
    @staticmethod
    def freshness(n_new, replay_size):
        return n_new / max(replay_size, 1)

    @staticmethod
    def policy_entropy(replay, sample_n=100):
        """Average entropy of policy targets in a sample."""
        if len(replay) < sample_n: return 0.0
        indices = random.sample(range(len(replay)), sample_n)
        total_ent = 0.0
        for i in indices:
            _, pol, _ = replay.buf[i]
            p = np.array(pol, dtype=np.float32)
            p = p[p > 1e-8]
            if len(p) > 0:
                total_ent += -np.sum(p * np.log(p))
        return total_ent / sample_n

    @staticmethod
    def value_std(replay, sample_n=100):
        """Std of value targets (diversity proxy)."""
        if len(replay) < sample_n: return 0.0
        indices = random.sample(range(len(replay)), sample_n)
        vals = [replay.buf[i][2] for i in indices]
        return float(np.std(vals))


class StepEarlyStopping:
    """Loose within-iteration plateau stopper.

    Unlike outer stopping, this only shortens excessively long inner epochs after
    a substantial fraction of planned steps has already run.
    """
    def __init__(self, patience=8, min_delta=5e-4, min_fraction=0.7, ema_alpha=0.2, planned_steps=1):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.min_fraction = float(max(0.0, min(1.0, min_fraction)))
        self.ema_alpha = float(max(0.01, min(1.0, ema_alpha)))
        self.planned_steps = max(1, int(planned_steps))
        self.min_steps = max(1, int(math.ceil(self.planned_steps * self.min_fraction)))
        self.best_loss = float("inf")
        self.loss_ema = None
        self.counter = 0
        self.triggered = False

    def step(self, loss, steps_done):
        loss = float(loss)
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            a = self.ema_alpha
            self.loss_ema = a * loss + (1.0 - a) * self.loss_ema

        if self.loss_ema < self.best_loss - self.min_delta:
            self.best_loss = self.loss_ema
            self.counter = 0
        else:
            self.counter += 1

        if steps_done < self.min_steps:
            return False
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered

    def summary(self, steps_done):
        return {
            "triggered": bool(self.triggered),
            "steps_done": int(steps_done),
            "min_steps": int(self.min_steps),
            "planned_steps": int(self.planned_steps),
            "counter": int(self.counter),
            "loss_ema": round_or_none(self.loss_ema),
        }


def train_epoch(model, optimizer, replay, cfg, device, n_steps, backend=None, inner_stop_cfg=None):
    """Train for n_steps. Uses backend.train_step if available (JAX JIT)."""
    total_loss, total_pl, total_vl = 0.0, 0.0, 0.0
    loader = replay.build_dataloader(
        cfg['batch'],
        n_steps,
        pin_memory=(backend is None and getattr(device, "type", "cpu") != "cpu"),
    )
    if loader is None:
        return 0.0, 0.0, 0.0, 0, None
    if backend is None:
        model.train()
    executed_steps = 0
    step_stopper = None
    if inner_stop_cfg and int(inner_stop_cfg.get("patience", 0) or 0) > 0:
        step_stopper = StepEarlyStopping(
            patience=inner_stop_cfg.get("patience", 8),
            min_delta=inner_stop_cfg.get("min_delta", 5e-4),
            min_fraction=inner_stop_cfg.get("min_fraction", 0.7),
            ema_alpha=inner_stop_cfg.get("ema_alpha", 0.2),
            planned_steps=n_steps,
        )

    with tqdm(loader, total=n_steps, desc="  Training", leave=False) as pbar:
        for states_t, policies_t, values_t in pbar:

            if backend is not None:
                # Backend path (JAX JIT or PyTorch via unified API)
                loss, pl, vl = backend.train_step(
                    states_t.numpy(), policies_t.numpy(), values_t.numpy())
            else:
                # Legacy PyTorch direct path
                states_t = states_t.to(device, non_blocking=True)
                policies_t = policies_t.to(device, non_blocking=True)
                values_t = values_t.to(device, non_blocking=True)
                logits, pred_v = model(states_t)
                log_probs = F.log_softmax(logits, dim=-1)
                pl = -(policies_t * log_probs).sum(dim=-1).mean()
                vl = F.mse_loss(pred_v, values_t)
                loss_t = pl + vl
                optimizer.zero_grad(set_to_none=True)
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss, pl, vl = loss_t.item(), pl.item(), vl.item()

            total_loss += loss; total_pl += pl; total_vl += vl
            executed_steps += 1
            pbar.set_postfix(loss=f"{loss:.3f}", p=f"{pl:.3f}", v=f"{vl:.3f}")
            if step_stopper and step_stopper.step(loss, executed_steps):
                break

    n = max(executed_steps, 1)
    return (
        total_loss / n,
        total_pl / n,
        total_vl / n,
        executed_steps,
        step_stopper.summary(executed_steps) if step_stopper is not None else None,
    )


_COMPILED_MODELS = {}  # id(model) → compiled model cache


def _get_compiled_model(model):
    """Lazily compile model with torch.compile for faster inference."""
    key = id(model)
    compiled = _COMPILED_MODELS.get(key)
    if compiled is not None:
        return compiled
    if os.environ.get("QUARTZ_DISABLE_COMPILE"):
        return model
    try:
        import torch as _torch
        compiled = _torch.compile(model, mode="default", dynamic=True)
        _COMPILED_MODELS[key] = compiled
        return compiled
    except Exception:
        _COMPILED_MODELS[key] = model
        return model


_PINNED_BUFS = {}  # (device_str, C, H, W) → (pinned_tensor, gpu_tensor, max_bs)


def _get_inference_buffers(device, batch_np):
    """Get or create pre-allocated pinned + GPU buffers for inference."""
    if batch_np.ndim != 4:
        return None
    bs, C, H, W = batch_np.shape
    key = (str(device), C, H, W)
    entry = _PINNED_BUFS.get(key)
    if entry is not None:
        pinned, gpu, max_bs = entry
        if bs <= max_bs:
            pinned[:bs].copy_(torch.from_numpy(batch_np))
            gpu[:bs].copy_(pinned[:bs], non_blocking=True)
            return gpu[:bs]
    # Allocate new buffers (2x headroom for batch size growth)
    max_bs = max(bs * 2, 64)
    pinned = torch.zeros(max_bs, C, H, W, dtype=torch.float32).pin_memory()
    gpu = torch.zeros(max_bs, C, H, W, dtype=torch.float32, device=device)
    _PINNED_BUFS[key] = (pinned, gpu, max_bs)
    pinned[:bs].copy_(torch.from_numpy(batch_np))
    gpu[:bs].copy_(pinned[:bs], non_blocking=True)
    return gpu[:bs]


def _run_model_batch(model, device, batch_features):
    batch_np = np.asarray(batch_features, dtype=np.float32)
    if not batch_np.flags.c_contiguous:
        batch_np = np.ascontiguousarray(batch_np)
    if not batch_np.flags.writeable:
        batch_np = batch_np.copy()
    if hasattr(model, "predict"):
        probs_batch, vals_np = model.predict(batch_np)
        return np.asarray(probs_batch, dtype=np.float32), np.asarray(vals_np, dtype=np.float32).reshape(-1)
    if getattr(device, "type", "cpu") != "cpu":
        x_batch = _get_inference_buffers(device, batch_np)
        if x_batch is None:
            x_batch = torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True)
    else:
        x_batch = torch.from_numpy(batch_np).to(device)
    compiled = _get_compiled_model(model)
    with torch.inference_mode():
        logits_batch, vals_batch = compiled(x_batch)
        probs_batch = torch.softmax(logits_batch, dim=-1).cpu().numpy()
        vals_np = vals_batch.cpu().numpy()
    return probs_batch, vals_np


def choose_selfplay_move(policy, legal, move_count, temp_threshold, fallback_best=-1):
    """Choose a self-play move from search policy with early-game exploration."""
    if not legal:
        return 0

    if move_count < temp_threshold:
        probs = np.array([policy[a] for a in legal], dtype=np.float64)
        total = probs.sum()
        if total > 1e-8:
            probs /= total
            return int(np.random.choice(legal, p=probs))
        return int(random.choice(legal))

    if fallback_best in legal:
        return int(fallback_best)
    return int(max(legal, key=lambda a: policy[a]))


def get_actor_model(training_model, backend):
    """Return the model object that self-play/eval should query for NN eval."""
    if backend is None:
        return training_model
    return backend


def clone_actor_model(actor_source):
    if actor_source is None:
        return None
    if hasattr(actor_source, "create_actor"):
        actor = actor_source.create_actor()
    else:
        import copy
        actor = copy.deepcopy(actor_source)
    if hasattr(actor, "eval"):
        actor.eval()
    return actor


def load_actor_source_from_checkpoint(
        checkpoint_path, cfg, device, backend_preference="torch", backend_template=None):
    """Load an inference actor from checkpoint using the matching backend."""
    backend_preference = str(backend_preference or "torch").lower()
    if backend_preference == "jax":
        if backend_template is not None and getattr(backend_template, "name", "") == "jax":
            return backend_template.load_actor(checkpoint_path)
        from quartz.backend import create_backend
        backend = create_backend(cfg, device="jax", preference="jax")
        if hasattr(backend, "load_actor"):
            return backend.load_actor(checkpoint_path)
        if not backend.load(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        return backend.create_actor()
    actor = AlphaZeroNet(cfg).to(device)
    actor.load_state_dict(load_torch_state_dict(checkpoint_path, torch, map_location=device))
    actor.eval()
    return actor


def wait_for_worker_progress(worker, previous_count, min_new=1, timeout_s=30.0, poll_s=0.25):
    """Wait for background self-play to add new positions to replay."""
    deadline = time.time() + timeout_s
    current = worker.positions_generated
    while current - previous_count < min_new and time.time() < deadline:
        if worker._stop.is_set():
            break
        status = worker.status() if hasattr(worker, "status") else None
        if status is not None:
            if not status.get("alive", True):
                raise RuntimeError(
                    f"background self-play worker stopped unexpectedly: {status.get('last_error') or 'thread exited'}"
                )
            if (
                status.get("last_progress_age_s", 0.0) > getattr(worker, "REPLAY_STALL_TIMEOUT_S", 45.0)
                and status.get("consecutive_errors", 0) > 0
            ):
                raise RuntimeError(
                    f"background self-play stalled: {status.get('last_error') or 'no progress'}"
                )
        time.sleep(poll_s)
        current = worker.positions_generated
    return max(0, current - previous_count), current


def compute_train_steps(base_steps, batch_size, n_new, concurrent=False):
    """Scale learner work to the amount of fresh data available."""
    if not concurrent:
        return base_steps
    if n_new <= 0:
        return 0
    target_reuse = 8.0
    scaled = math.ceil((n_new / max(batch_size, 1)) * target_reuse)
    return max(1, min(base_steps, scaled))


def default_output_dir(game_name):
    """Persist training artifacts under the project-local models directory."""
    return os.path.join("models", f"alphazero_{game_name}")


@dataclass
class HardwareSpec:
    logical_cpus: int
    physical_cpus: int
    memory_mb: int
    gpu_vendor: str = "none"
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    gpu_count: int = 0
    torch_cuda: bool = False
    device_kind: str = "cpu"


def _detect_cpu_counts():
    logical = 0
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1
    logical = max(1, logical)

    physical = 0
    if sys.platform.startswith("linux"):
        try:
            pairs = set()
            current_phys = None
            current_core = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        current_phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        current_core = line.split(":", 1)[1].strip()
                    elif not line.strip():
                        if current_phys is not None and current_core is not None:
                            pairs.add((current_phys, current_core))
                        current_phys = None
                        current_core = None
            physical = len(pairs)
        except Exception:
            physical = 0
    if physical <= 0:
        physical = max(1, logical // 2)
    return logical, physical


def _detect_memory_mb():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
    return 0


def detect_hardware_spec(device):
    logical, physical = _detect_cpu_counts()
    memory_mb = _detect_memory_mb()

    gpu_vendor = "none"
    gpu_name = ""
    gpu_vram_mb = 0
    gpu_count = 0
    torch_cuda = bool(torch.cuda.is_available())
    device_kind = getattr(device, "type", str(device))

    if detect_gpu is not None:
        try:
            gpu_info = detect_gpu()
            gpu_vendor = gpu_info.vendor or gpu_vendor
            gpu_name = gpu_info.device_name or gpu_name
            gpu_vram_mb = gpu_info.vram_mb or gpu_vram_mb
        except Exception:
            pass

    if torch_cuda:
        try:
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gpu_name = props.name or gpu_name
            gpu_vram_mb = max(gpu_vram_mb, int(props.total_memory // (1024 * 1024)))
            if gpu_vendor == "none":
                gpu_vendor = "cuda"
        except Exception:
            gpu_count = max(gpu_count, 1)

    return HardwareSpec(
        logical_cpus=logical,
        physical_cpus=physical,
        memory_mb=memory_mb,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_count=gpu_count,
        torch_cuda=torch_cuda,
        device_kind=device_kind,
    )


def configure_torch_rocm_runtime(hw):
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return
    gpu_name = (hw.gpu_name or "").lower()
    unsupported_lt = (
        "gfx1030" in gpu_name
        or "rx 6950" in gpu_name
        or "rx 6900" in gpu_name
        or "rx 6800" in gpu_name
    )
    if not unsupported_lt:
        return
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if preferred_blas is None:
        return
    try:
        preferred_blas("hipblas")
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
        category=UserWarning,
    )


def recommend_eval_parallel_workers(hw, cfg, eval_games, rust_ok):
    if eval_games <= 1:
        return 1
    thread_cost = max(1, int(cfg.get("n_threads", 1)))
    cpu_capacity = max(1, hw.physical_cpus // thread_cost)
    return max(1, min(cpu_capacity, int(eval_games)))


EVAL_AUTOTUNE_PROFILE_VERSION = 4


def eval_autotune_signature(hw, cfg, eval_games):
    return {
        "hardware": hardware_signature(hw),
        "game": cfg.get("_name"),
        "eval_games": int(eval_games),
        "iters": int(cfg.get("iters", 0)),
        "n_threads": int(cfg.get("n_threads", 1)),
        "batch_size": int(cfg.get("batch_size", 8)),
        "backend": str(cfg.get("_backend_name", "torch")),
        "search_profile": str(cfg.get("search_profile", "quartz")),
        "penalty_mode": str(cfg.get("penalty_mode", "GatedRefresh")),
        "batch_timeout_us": int(cfg.get("batch_timeout_us", 0) or 0),
        "eval_runner_mode": str(cfg.get("_eval_runner_mode", "python_batched")),
        "shared_eval_session": bool(cfg.get("_shared_eval_session", False)),
        "broker_enabled": bool(cfg.get("_broker_enabled", False)),
        "eval_topology_version": 4,
    }


def load_eval_autotune_profile(profile_path, hw, cfg, eval_games):
    if not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path) as f:
            data = json.load(f)
        if data.get("version") != EVAL_AUTOTUNE_PROFILE_VERSION:
            return None
        if data.get("signature") != eval_autotune_signature(hw, cfg, eval_games):
            return None
        workers = int(data.get("workers", 0) or 0)
        return workers if workers > 0 else None
    except Exception:
        return None


def save_eval_autotune_profile(profile_path, hw, cfg, eval_games, workers, benchmarks):
    payload = {
        "version": EVAL_AUTOTUNE_PROFILE_VERSION,
        "signature": eval_autotune_signature(hw, cfg, eval_games),
        "workers": int(workers),
        "benchmarks": benchmarks,
        "saved_at": int(time.time()),
    }
    with open(profile_path, "w") as f:
        json.dump(payload, f, indent=2)


def eval_worker_candidates(hw, cfg, eval_games):
    thread_cost = max(1, int(cfg.get("n_threads", 1)))
    cap = max(1, min(int(eval_games), hw.physical_cpus // thread_cost))
    seeds = [
        1,
        2,
        3,
        max(1, cap // 2),
        max(1, (cap * 2) // 3),
        cap,
    ]
    return [w for w in sorted(set(int(x) for x in seeds)) if 1 <= w <= cap]


def compute_eval_collect_policy(base_target_items, base_timeout_s, batch_items_ema=None, wait_ema_s=None):
    target = max(1, int(base_target_items))
    timeout_s = max(0.0005, float(base_timeout_s))
    items_ema = float(batch_items_ema if batch_items_ema is not None else target)
    wait_ema_s = max(0.0, float(wait_ema_s or 0.0))
    fill_ratio = max(0.0, min(2.0, items_ema / max(float(target), 1.0)))

    if fill_ratio < 0.55:
        timeout_s *= min(4.0, 1.0 / max(fill_ratio, 0.25))
    elif fill_ratio > 0.90:
        timeout_s *= 0.85
        target = min(64, max(target, int(round(items_ema * 1.15))))

    if wait_ema_s > timeout_s * 1.25:
        timeout_s *= 0.8
        target = max(1, min(target, int(round(max(1.0, items_ema)))))

    timeout_s = min(0.02, max(0.0005, timeout_s))
    return target, timeout_s


def benchmark_eval_parallel_workers(
        hw, cfg, eval_games, candidate_factory, champion_factory, game_factory, profile_path):
    if not HAS_EVAL_SYSTEM or eval_games <= 1:
        return 1, []

    candidates = eval_worker_candidates(hw, cfg, eval_games)
    if len(candidates) == 1:
        return candidates[0], []

    pilot_games = max(4, min(8, int(eval_games)))
    benchmarks = []
    best_workers = candidates[0]
    best_score = -1.0

    for workers in candidates:
        bench_cfg = EvalConfig(
            num_games=pilot_games,
            promotion_threshold=0.55,
            confidence=0.95,
            sanity_check_interval=10**9,
            sanity_games=0,
            color_swap=True,
            max_moves=cfg.get("max_moves", 500),
            seed=cfg.get("seed", None),
            parallel_workers=workers,
        )
        evaluator = TrainingEvaluator(config=bench_cfg)
        candidate = candidate_factory()
        champion = champion_factory()
        try:
            t0 = time.time()
            result = evaluator.evaluate_checkpoint(
                candidate=candidate,
                champion=champion,
                game_factory=game_factory,
                candidate_id="eval_autotune",
                generation=0,
                candidate_factory=candidate_factory,
                champion_factory=champion_factory,
            )
            elapsed = max(time.time() - t0, 1e-6)
            scored_games = int(result.tally.get("scored", 0) if result.tally else 0)
            total_games = int(result.conditions.get("num_games", pilot_games))
            games_per_s = float(total_games) / elapsed
            score = games_per_s * (1.0 if scored_games > 0 else 0.5)
            row = {
                "workers": workers,
                "games": total_games,
                "elapsed_s": round(elapsed, 3),
                "games_per_s": round(games_per_s, 4),
                "scored_games": scored_games,
                "score": round(score, 4),
            }
            benchmarks.append(row)
            if score > best_score:
                best_score = score
                best_workers = workers
        except Exception as e:
            benchmarks.append({
                "workers": workers,
                "error": str(e),
            })
        finally:
            try:
                candidate.reset()
            except Exception:
                pass
            try:
                champion.reset()
            except Exception:
                pass

    save_eval_autotune_profile(profile_path, hw, cfg, eval_games, best_workers, benchmarks)
    return best_workers, benchmarks


def _round_down_to_multiple(value, multiple):
    if multiple <= 1:
        return value
    return max(multiple, (value // multiple) * multiple)


def _round_up_to_multiple(value, multiple):
    if multiple <= 1:
        return value
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def _autotune_parallel_limit(hw, concurrent):
    logical = max(1, int(hw.logical_cpus))
    physical = max(1, int(hw.physical_cpus))
    upper = max(1, min(physical, logical))
    if concurrent and hw.device_kind != "cpu" and hw.gpu_vram_mb > 0:
        # Favor fewer Rust processes and more threads per process on GPU-backed
        # self-play to reduce Python<->Rust QIPC round-trips.
        ipc_cap = max(4, (physical + 1) // 2)
        upper = min(upper, ipc_cap)
    return max(1, upper)


def _autotune_thread_capacity(hw, parallel):
    logical = max(1, int(hw.logical_cpus))
    capacity_basis = logical
    if hw.device_kind != "cpu" and hw.gpu_vram_mb > 0:
        # Resident Rust sessions reduce IPC, but very high per-process thread
        # counts still tend to produce long self-play cycles and poor wall-clock
        # responsiveness on GPU-backed training. Keep the search fan-out bounded.
        gpu_cap = 6 if parallel <= 2 else 4 if parallel <= 4 else 3
    else:
        gpu_cap = None
        # CPU autotune stays conservative by sizing search threads against
        # physical cores, while the hard ceiling is still the logical thread
        # count reported by the machine.
        capacity_basis = max(1, int(hw.physical_cpus))
    if capacity_basis <= 8:
        cap = max(1, min(8, capacity_basis // max(1, int(parallel)) or 1))
        cap = min(cap, logical)
        return min(cap, gpu_cap) if gpu_cap is not None else cap
    reserve = 1 if capacity_basis >= 12 else 0
    usable = max(1, capacity_basis - reserve)
    cap = max(1, min(12, usable // max(1, int(parallel))))
    cap = min(cap, logical)
    return min(cap, gpu_cap) if gpu_cap is not None else cap


def _autotune_thread_candidates(hw, parallel, hinted=None):
    cap = _autotune_thread_capacity(hw, parallel)
    if cap <= 1:
        seeds = [1]
    elif hw.device_kind != "cpu" and hw.gpu_vram_mb > 0 and parallel >= 4:
        seeds = [2, 3, min(4, cap), cap]
    else:
        seeds = [1, 2, 3, min(4, cap), cap]
    if hinted is not None:
        seeds.append(min(cap, max(1, int(hinted))))
    return [t for t in sorted(set(int(x) for x in seeds)) if 1 <= t <= cap]


def _autotune_batch_game_limit(hw, parallel, concurrent):
    parallel = max(1, int(parallel))
    if not concurrent:
        return parallel
    physical = max(1, int(hw.physical_cpus))
    return max(parallel, min(physical * 2, parallel * 2))


def estimate_model_params(cfg):
    model = AlphaZeroNet(cfg)
    return sum(p.numel() for p in model.parameters())


def autoscale_model_cfg(cfg, hw):
    tuned = dict(cfg)
    current_params = estimate_model_params(tuned)
    board_area = cfg["board"] * cfg["board"]
    actions = cfg["actions"]

    if hw.device_kind == "cpu" or hw.gpu_vram_mb <= 0:
        return tuned

    if board_area <= 64 and actions <= 128:
        target_floor = 1_000_000 if hw.gpu_vram_mb >= 12_000 else 750_000
    elif board_area <= 100 and actions <= 256:
        target_floor = 2_000_000 if hw.gpu_vram_mb >= 12_000 else 1_500_000
    else:
        target_floor = 3_500_000 if hw.gpu_vram_mb >= 12_000 else 2_500_000

    if hw.gpu_vram_mb >= 20_000:
        max_params_cap = 8_000_000
    elif hw.gpu_vram_mb >= 16_000:
        max_params_cap = 6_000_000
    elif hw.gpu_vram_mb >= 12_000:
        max_params_cap = 4_500_000
    elif hw.gpu_vram_mb >= 8_000:
        max_params_cap = 3_000_000
    else:
        max_params_cap = 1_500_000

    if current_params >= int(target_floor * 0.9):
        return tuned

    base_filters = cfg["filters"]
    base_blocks = cfg["blocks"]
    base_vh = cfg["vh"]
    filter_cap = max(base_filters, 192 if board_area <= 64 else 256)
    block_cap = max(base_blocks, 8 if board_area <= 64 else 12)
    vh_cap = 512 if hw.gpu_vram_mb >= 12_000 else 256

    filter_values = sorted(set([
        base_filters,
        _round_up_to_multiple(base_filters + 32, 32),
        _round_up_to_multiple(base_filters + 64, 32),
        filter_cap,
    ]))
    filter_values = [f for f in filter_values if base_filters <= f <= filter_cap]

    block_values = list(range(base_blocks, block_cap + 1, 2))
    vh_values = sorted(set([
        base_vh,
        _round_up_to_multiple(max(base_vh, base_filters), 64),
        _round_up_to_multiple(max(base_vh, base_filters * 2), 64),
        vh_cap,
    ]))
    vh_values = [v for v in vh_values if base_vh <= v <= vh_cap]

    candidates = []
    for filters in filter_values:
        for blocks in block_values:
            for vh in vh_values:
                candidate = dict(tuned)
                candidate["filters"] = filters
                candidate["blocks"] = blocks
                candidate["vh"] = vh
                params = estimate_model_params(candidate)
                if params < current_params or params > max_params_cap:
                    continue
                candidates.append((params, filters, blocks, vh))

    if not candidates:
        return tuned

    above_floor = [c for c in candidates if c[0] >= target_floor]
    if above_floor:
        _, filters, blocks, vh = min(above_floor, key=lambda c: c[0])
    else:
        _, filters, blocks, vh = max(candidates, key=lambda c: c[0])

    if (filters, blocks, vh) != (base_filters, base_blocks, base_vh):
        tuned["filters"] = filters
        tuned["blocks"] = blocks
        tuned["vh"] = vh
    return tuned


def _probe_inference_batch_size(model, device, cfg, eval_batch_cap):
    """Benchmark different batch sizes to find optimal GPU throughput."""
    if model is None or hasattr(model, "predict") or getattr(device, "type", "cpu") == "cpu":
        return cfg.get("batch_size", 8)
    ch, bs = cfg.get("ch", 3), cfg.get("board", 7)
    current_bs = cfg.get("batch_size", 8)
    candidates = sorted(set([current_bs] + [c for c in [32, 64, 128, 256] if c <= eval_batch_cap]))
    best_bs, best_ips = current_bs, 0.0
    import time as _time
    for cand in candidates:
        try:
            batch = [np.random.randn(ch, bs, bs).astype(np.float32) for _ in range(cand)]
            # Warmup
            for _ in range(5):
                _run_model_batch(model, device, batch)
            N = max(20, 200 // cand)
            t0 = _time.perf_counter()
            for _ in range(N):
                _run_model_batch(model, device, batch)
            elapsed = _time.perf_counter() - t0
            ips = cand * N / max(elapsed, 1e-9)
            if ips > best_ips:
                best_ips = ips
                best_bs = cand
        except Exception:
            break  # OOM or other error — stop probing larger sizes
    return best_bs


def autotune_training_cfg(cfg, hw, concurrent=True):
    tuned = autoscale_model_cfg(cfg, hw)

    proc_target = _autotune_parallel_limit(hw, concurrent)

    if hw.gpu_vram_mb >= 20_000:
        train_batch_scale = 2.0
        eval_batch_cap = 256
    elif hw.gpu_vram_mb >= 16_000:
        train_batch_scale = 1.5
        eval_batch_cap = 192
    elif hw.gpu_vram_mb >= 12_000:
        train_batch_scale = 1.5
        eval_batch_cap = 128
    elif hw.gpu_vram_mb >= 8_000:
        train_batch_scale = 1.0
        eval_batch_cap = 64
    elif hw.gpu_vram_mb >= 4_000:
        train_batch_scale = 0.75
        eval_batch_cap = 32
    else:
        train_batch_scale = 0.5 if hw.device_kind == "cpu" else 0.75
        eval_batch_cap = 16

    max_parallel = proc_target
    tuned["selfplay_parallel"] = max(1, min(max_parallel, cfg.get("games", 1)))
    tuned["bg_parallel"] = max(1, min(max_parallel, cfg.get("games", max_parallel)))
    parallel_den = max(tuned["selfplay_parallel"], tuned["bg_parallel"], 1)
    tuned["n_threads"] = _autotune_thread_capacity(hw, parallel_den)
    tuned["batch_size"] = max(
        cfg.get("batch_size", 8),
        min(eval_batch_cap, max(tuned["n_threads"] * tuned["bg_parallel"], 8)))

    base_batch = cfg.get("batch", 256)
    batch_multiple = 32 if base_batch >= 256 else 16
    tuned["batch"] = _round_down_to_multiple(
        int(base_batch * train_batch_scale), batch_multiple)
    tuned["batch"] = max(batch_multiple, tuned["batch"])

    if concurrent:
        tuned["bg_batch_games"] = _autotune_batch_game_limit(hw, tuned["bg_parallel"], concurrent=True)
    else:
        tuned["bg_batch_games"] = 0

    if not concurrent:
        if hw.logical_cpus >= 24 and hw.gpu_vram_mb >= 12_000:
            tuned["games"] = max(cfg.get("games", 1), tuned["selfplay_parallel"] * 60)
        elif hw.logical_cpus >= 12:
            tuned["games"] = max(cfg.get("games", 1), tuned["selfplay_parallel"] * 50)

    tuned["hw_logical_cpus"] = hw.logical_cpus
    tuned["hw_memory_mb"] = hw.memory_mb
    tuned["hw_gpu_vram_mb"] = hw.gpu_vram_mb
    return tuned


def print_autotune_summary(original_cfg, tuned_cfg, hw):
    print("  Hardware:")
    print(f"    CPU: {hw.logical_cpus} logical / {hw.physical_cpus} physical cores")
    if hw.memory_mb:
        print(f"    RAM: {hw.memory_mb:,} MB")
    if hw.gpu_name or hw.gpu_vendor != "none":
        gpu_desc = hw.gpu_name or hw.gpu_vendor
        if hw.gpu_vram_mb:
            gpu_desc += f" ({hw.gpu_vram_mb:,} MB)"
        print(f"    GPU: {gpu_desc}")

    changed = []
    for key in ("filters", "blocks", "vh", "games", "batch", "n_threads", "batch_size",
                "selfplay_parallel", "bg_parallel", "bg_batch_games"):
        if original_cfg.get(key) != tuned_cfg.get(key):
            changed.append(f"{key}={tuned_cfg.get(key)}")
    if changed:
        print("  Auto-tuned:", ", ".join(changed))
    else:
        print("  Auto-tuned: no changes")


def hardware_signature(hw):
    return {
        "logical_cpus": hw.logical_cpus,
        "physical_cpus": hw.physical_cpus,
        "memory_mb": hw.memory_mb,
        "gpu_vendor": hw.gpu_vendor,
        "gpu_name": hw.gpu_name,
        "gpu_vram_mb": hw.gpu_vram_mb,
        "device_kind": hw.device_kind,
    }


def max_supported_threads(hw):
    return max(1, int(getattr(hw, "logical_cpus", 1) or 1))


def gpu_host_thread_cap(hw):
    logical = max(1, int(getattr(hw, "logical_cpus", 1) or 1))
    physical = max(1, int(getattr(hw, "physical_cpus", logical) or logical))
    return max(1, min(logical, physical))


def gpu_interop_thread_cap(hw):
    logical = max(1, int(getattr(hw, "logical_cpus", 1) or 1))
    physical = max(1, int(getattr(hw, "physical_cpus", logical) or logical))
    return max(1, min(logical, max(1, physical // 2)))


def auto_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if sys.platform == "darwin" and mps_backend is not None:
        try:
            if bool(mps_backend.is_available()):
                return "mps"
        except Exception:
            pass
    return "cpu"


def clamp_thread_count(value, hw):
    return max(1, min(int(value), max_supported_threads(hw)))


def clamp_runtime_cfg_to_hardware(cfg, hw):
    out = dict(cfg)
    thread_cap = max_supported_threads(hw)
    if "n_threads" in out:
        out["n_threads"] = max(1, min(int(out["n_threads"]), thread_cap))
    return out


AUTOTUNE_PROFILE_VERSION = 16


def autotune_signature(hw, cfg):
    return {
        "hardware": hardware_signature(hw),
        "game": cfg.get("_name"),
        "iters": int(cfg.get("iters", 0)),
        "search_profile": str(cfg.get("search_profile", "quartz")),
        "penalty_mode": str(cfg.get("penalty_mode", "GatedRefresh")),
        "batch_timeout_us": int(cfg.get("batch_timeout_us", 0) or 0),
        "selfplay_topology_version": int(cfg.get("_selfplay_topology_version", 4)),
        "resident_session": bool(cfg.get("_resident_session", False)),
        "shared_eval_session": bool(cfg.get("_shared_eval_session", False)),
        "selfplay_runner_mode": str(cfg.get("_selfplay_runner_mode", "python_batched")),
        "autotune_topology_version": 5,
    }


def load_autotune_profile(profile_path, hw, cfg):
    if not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path) as f:
            data = json.load(f)
        if data.get("version") != AUTOTUNE_PROFILE_VERSION:
            return None
        if data.get("signature") != autotune_signature(hw, cfg):
            return None
        return data
    except Exception:
        return None


def save_autotune_profile(profile_path, hw, cfg, overrides, benchmarks):
    payload = {
        "version": AUTOTUNE_PROFILE_VERSION,
        "signature": autotune_signature(hw, cfg),
        "overrides": overrides,
        "benchmarks": benchmarks,
        "saved_at": int(time.time()),
    }
    with open(profile_path, "w") as f:
        json.dump(payload, f, indent=2)


def apply_runtime_overrides(cfg, overrides):
    out = dict(cfg)
    for key, value in (overrides or {}).items():
        out[key] = value
    return out


def _autotune_parallel_candidates(cfg, hw, concurrent):
    upper = _autotune_parallel_limit(hw, concurrent)
    if concurrent:
        seeds = [
            2,
            3,
            min(4, upper),
            max(1, upper // 2),
            max(1, (upper * 2) // 3),
            cfg.get("bg_parallel", 1),
            upper,
        ]
        if hw.device_kind == "cpu" or hw.gpu_vram_mb <= 0:
            seeds.insert(0, 1)
        if upper > 4:
            seeds.extend([max(1, upper - 1), max(1, upper - 2)])
    else:
        hinted = max(1, cfg.get("selfplay_parallel", 1))
        seeds = [1, 2, 3, min(4, upper), hinted, min(upper, hinted + 2), upper]
    return [p for p in sorted(set(int(x) for x in seeds)) if 1 <= p <= upper]


def _autotune_batch_game_candidates(hw, parallel, concurrent):
    if not concurrent:
        return [parallel]
    cap = _autotune_batch_game_limit(hw, parallel, concurrent)
    return sorted(set([
        max(1, parallel),
        max(1, cap),
    ]))


def _score_selfplay_probe(positions_per_s, cycle_s, concurrent, positions=0, eval_messages=0,
                          model_batch_mean=0.0, parallel=1, n_threads=1):
    if not concurrent:
        return positions_per_s
    score = positions_per_s / math.sqrt(max(cycle_s, 1e-6))
    if positions > 0 and eval_messages > 0:
        message_efficiency = max(float(positions) / float(eval_messages), 1e-6)
        score *= message_efficiency ** 0.25
    if model_batch_mean > 0.0:
        # Reward actual cross-game batching. Batch mean near 1.0 indicates
        # callback flood and poor GPU utilization, even if raw pos/s looks okay.
        score *= max(float(model_batch_mean), 1e-6) ** 0.35
    if parallel > 1 and n_threads <= 1 and model_batch_mean <= 1.25:
        # Multi-game + single-thread + no batching has consistently been a bad
        # shape in repro runs; penalize heavily so coarse autotune does not
        # overfit to callback-heavy configurations.
        score *= 0.35
    return score


def _sync_device(device):
    if getattr(device, "type", "cpu") != "cpu" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass


def _mean_or_zero(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _warmup_selfplay_probe(cfg, model, device, rust_binary, parallel, batch_games, n_threads):
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = n_threads
    probe_cfg["batch_size"] = max(
        cfg.get("batch_size", 8),
        min(64, max(n_threads * parallel, 8)))
    probe_cfg["_disable_resident_session"] = True
    if model is not None:
        ch, bs = probe_cfg["ch"], probe_cfg["board"]
        warm_batch = np.zeros((probe_cfg["batch_size"], ch, bs, bs), dtype=np.float32)
        _run_model_batch(model, device, warm_batch)
        _sync_device(device)
    warm_cfg = dict(probe_cfg)
    warm_cfg["iters"] = min(8, max(4, int(cfg.get("iters", 8))))
    try:
        selfplay_rust_nn_batched(
            warm_cfg, model, device, max(1, batch_games), rust_binary,
            parallel=parallel, show_progress=False)
    except Exception:
        pass
    _sync_device(device)


def _run_selfplay_probe(cfg, model, device, rust_binary, parallel, batch_games, n_threads,
                        concurrent=True, rounds=1, warmup=True):
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = n_threads
    probe_cfg["batch_size"] = max(
        cfg.get("batch_size", 8),
        min(64, max(n_threads * parallel, 8)))
    probe_cfg["_disable_resident_session"] = True
    if warmup:
        _warmup_selfplay_probe(cfg, model, device, rust_binary, parallel, batch_games, n_threads)
    n_games = max(1, batch_games * max(1, rounds))
    perf_stats = {}
    _stall_trace(
        "selfplay_probe_begin",
        game=cfg.get("_name"),
        parallel=int(parallel),
        batch_games=int(batch_games),
        n_threads=int(n_threads),
        rounds=int(rounds),
        n_games=int(n_games),
        iters=int(probe_cfg.get("iters", 0)),
        batch_size=int(probe_cfg.get("batch_size", 0)),
        resident=bool(probe_cfg.get("_resident_session", False)),
    )
    _sync_device(device)
    t0 = time.time()
    states, _, _, _ = selfplay_rust_nn_batched(
        probe_cfg, model, device, n_games, rust_binary,
        parallel=parallel, show_progress=False, perf_stats=perf_stats)
    _sync_device(device)
    elapsed = max(time.time() - t0, 1e-6)
    positions = sum(len(gs) for gs in states)
    cycle_s = elapsed / max(rounds, 1)
    positions_per_s = positions / elapsed
    positions_per_cycle = positions / max(rounds, 1)
    eval_messages = int(perf_stats.get("eval_messages", 0) or 0)
    model_batch_mean = _mean_or_zero(perf_stats.get("model_batch_sizes", []))
    score = _score_selfplay_probe(
        positions_per_s,
        cycle_s,
        concurrent,
        positions=positions,
        eval_messages=eval_messages,
        model_batch_mean=model_batch_mean,
        parallel=parallel,
        n_threads=n_threads,
    )
    _stall_trace(
        "selfplay_probe_end",
        game=cfg.get("_name"),
        parallel=int(parallel),
        batch_games=int(batch_games),
        n_threads=int(n_threads),
        rounds=int(rounds),
        positions=int(positions),
        elapsed_s=float(elapsed),
        cycle_s=float(cycle_s),
        positions_per_s=float(positions_per_s),
        eval_messages=int(eval_messages),
        model_calls=int(perf_stats.get("model_calls", 0) or 0),
    )
    return {
        "parallel": parallel,
        "batch_games": batch_games,
        "n_threads": n_threads,
        "batch_size": probe_cfg["batch_size"],
        "probe_rounds": rounds,
        "positions": positions,
        "elapsed_s": round(elapsed, 3),
        "cycle_s": round(cycle_s, 3),
        "positions_per_cycle": round(positions_per_cycle, 3),
        "positions_per_s": round(positions_per_s, 3),
        "eval_messages": eval_messages,
        "eval_items": int(perf_stats.get("eval_items", 0) or 0),
        "model_calls": int(perf_stats.get("model_calls", 0) or 0),
        "model_batch_mean": round(model_batch_mean, 3),
        "score": round(score, 4),
    }


def should_use_resident_session(game_name, parallel, n_games, enabled=False):
    """Resident sessions help when multiple concurrent games share one Rust server.

    For single-game probes they add session lifecycle complexity with no batching
    upside, and in practice can stall the first autotune candidate.

    Chess support exists, but auto-enable policy remains more conservative and is
    decided separately from this capability check.
    """
    if not enabled:
        return False
    return int(parallel) > 1 and int(n_games) > 1


def supports_rust_eval_state_machine(game_name):
    rg = rust_game_name(game_name)
    return rg in GAME_CONFIGS or rg in GOMOKU15_VARIANTS or is_chess_game(rg) or is_go_game(rg)


def supports_rust_selfplay_state_machine(game_name):
    rg = rust_game_name(game_name)
    return rg in GAME_CONFIGS or rg in GOMOKU15_VARIANTS or is_chess_game(rg) or is_go_game(rg)


def _score_train_batch_probe(examples_per_s, batch_n, concurrent=False, target_positions_per_cycle=None):
    if not concurrent or not target_positions_per_cycle:
        return examples_per_s
    target_batch = max(32.0, float(target_positions_per_cycle) * 4.0)
    freshness_penalty = min(1.0, target_batch / max(float(batch_n), 1.0))
    return examples_per_s * freshness_penalty


def _autotune_progress_bar(total, desc):
    use_tqdm = HAS_TQDM and sys.stderr.isatty()
    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True, disable=not use_tqdm)


def benchmark_selfplay_throughput(cfg, model, device, rust_binary, hw, concurrent=True):
    candidates = _autotune_parallel_candidates(cfg, hw, concurrent)
    coarse_cfg = dict(cfg)
    coarse_cfg["iters"] = min(cfg["iters"], 48 if cfg["board"] <= 9 else 32)
    coarse_cfg["temp_th"] = min(cfg["temp_th"], 6)
    refine_cfg = dict(cfg)
    refine_cfg["iters"] = min(
        cfg["iters"],
        128 if cfg["board"] <= 9 else 80)
    refine_cfg["temp_th"] = min(cfg["temp_th"], 8)

    coarse_plan = []
    for parallel in candidates:
        for batch_games in _autotune_batch_game_candidates(hw, parallel, concurrent):
            thread_candidates = _autotune_thread_candidates(
                hw, parallel, hinted=cfg.get("n_threads", 1))
            for n_threads in thread_candidates:
                coarse_plan.append((parallel, batch_games, n_threads))

    results = []
    with _autotune_progress_bar(len(coarse_plan), "autotune-selfplay-coarse") as pbar:
        for parallel, batch_games, n_threads in coarse_plan:
            pbar.set_postfix_str(f"p={parallel} bg={batch_games} th={n_threads}")
            try:
                probe = _run_selfplay_probe(
                    coarse_cfg, model, device, rust_binary,
                    parallel=parallel, batch_games=batch_games, n_threads=n_threads,
                    concurrent=concurrent, rounds=1 if concurrent else 1)
            except Exception as e:
                results.append({
                    "parallel": parallel,
                    "batch_games": batch_games,
                    "n_threads": n_threads,
                    "error": str(e),
                })
                pbar.update(1)
                continue
            probe["stage"] = "coarse"
            results.append(probe)
            pbar.update(1)

    scored = [r for r in results if "positions_per_s" in r]
    if not scored:
        return {}, results, {}

    finalists = []
    seen = set()
    for row in sorted(scored, key=lambda r: (r["score"], r["positions_per_s"], -r["cycle_s"]), reverse=True):
        key = (row["parallel"], row["batch_games"], row["n_threads"])
        if key in seen:
            continue
        seen.add(key)
        finalists.append(row)
        if len(finalists) >= min(4, len(scored)):
            break

    refined = []
    with _autotune_progress_bar(len(finalists), "autotune-selfplay-refine") as pbar:
        for row in finalists:
            pbar.set_postfix_str(
                f"p={row['parallel']} bg={row['batch_games']} th={row['n_threads']}"
            )
            try:
                probe = _run_selfplay_probe(
                    refine_cfg, model, device, rust_binary,
                    parallel=row["parallel"], batch_games=row["batch_games"], n_threads=row["n_threads"],
                    concurrent=concurrent, rounds=3 if concurrent else 1)
                probe["stage"] = "refine"
                refined.append(probe)
                results.append(probe)
            except Exception as e:
                results.append({
                    "parallel": row["parallel"],
                    "batch_games": row["batch_games"],
                    "n_threads": row["n_threads"],
                    "stage": "refine",
                    "error": str(e),
                })
            pbar.update(1)

    ranking_pool = refined or scored
    best = max(ranking_pool, key=lambda r: (r["score"], r["positions_per_s"], -r["cycle_s"]))
    overrides = {
        "selfplay_parallel": best["parallel"],
        "bg_parallel": best["parallel"] if concurrent else min(best["parallel"], 4),
        "bg_batch_games": best["batch_games"] if concurrent else max(1, best["parallel"]),
        "n_threads": best["n_threads"],
    }
    summary = {
        "parallel": best["parallel"],
        "batch_games": best["batch_games"],
        "n_threads": best["n_threads"],
        "positions_per_cycle": best["positions_per_cycle"],
        "positions_per_s": best["positions_per_s"],
        "cycle_s": best["cycle_s"],
        "score": best["score"],
    }
    return overrides, results, summary


def benchmark_train_batch(cfg, backend, model, optimizer, device, hw,
                          concurrent=False, target_positions_per_cycle=None):
    base_batch = cfg["batch"]
    batch_multiple = 32 if base_batch >= 256 else 16
    batch_candidates = sorted(set([
        _round_down_to_multiple(int(base_batch * 0.5), batch_multiple),
        _round_down_to_multiple(int(base_batch * 0.75), batch_multiple),
        base_batch,
        _round_down_to_multiple(int(base_batch * 1.25), batch_multiple),
        _round_down_to_multiple(int(base_batch * 1.5), batch_multiple),
    ]))
    batch_candidates = [b for b in batch_candidates if b >= 32]
    if concurrent and target_positions_per_cycle:
        max_useful_batch = _round_down_to_multiple(
            int(max(32, target_positions_per_cycle * 4.0)), batch_multiple)
        constrained = [b for b in batch_candidates if b <= max_useful_batch]
        if constrained:
            batch_candidates = constrained

    results = []
    ch, bs, actions = cfg["ch"], cfg["board"], cfg["actions"]
    import copy
    with _autotune_progress_bar(len(batch_candidates), "autotune-train-batch") as pbar:
        for batch_n in batch_candidates:
            pbar.set_postfix_str(f"batch={batch_n}")
            states = np.zeros((batch_n, ch, bs, bs), dtype=np.float32)
            policies = np.full((batch_n, actions), 1.0 / actions, dtype=np.float32)
            values = np.zeros(batch_n, dtype=np.float32)

            if backend is not None:
                if hasattr(backend, "optimizer"):
                    model_ref = backend.get_torch_model()
                    model_state = copy.deepcopy(model_ref.state_dict())
                    opt_state = copy.deepcopy(backend.optimizer.state_dict())

                    def restore():
                        model_ref.load_state_dict(model_state)
                        backend.optimizer.load_state_dict(opt_state)

                else:
                    params_state = copy.deepcopy(getattr(backend, "params", None))
                    batch_stats_state = copy.deepcopy(getattr(backend, "batch_stats", None))
                    opt_state_state = copy.deepcopy(getattr(backend, "opt_state", None))

                    def restore():
                        backend.params = copy.deepcopy(params_state)
                        backend.batch_stats = copy.deepcopy(batch_stats_state)
                        backend.opt_state = copy.deepcopy(opt_state_state)

                train_once = lambda: backend.train_step(states, policies, values)
            else:
                model_state = copy.deepcopy(model.state_dict())
                opt_state = copy.deepcopy(optimizer.state_dict())

                def train_once():
                    model.train()
                    states_t = torch.tensor(states, dtype=torch.float32).to(device)
                    policies_t = torch.tensor(policies, dtype=torch.float32).to(device)
                    values_t = torch.tensor(values, dtype=torch.float32).to(device)
                    logits, pred_v = model(states_t)
                    pl = -(policies_t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                    vl = F.mse_loss(pred_v, values_t)
                    loss_t = pl + vl
                    optimizer.zero_grad()
                    loss_t.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    return float(loss_t.item())

                restore = lambda: (model.load_state_dict(model_state), optimizer.load_state_dict(opt_state))

            try:
                train_once()
                t0 = time.time()
                measured = 2
                for _ in range(measured):
                    train_once()
                elapsed = max(time.time() - t0, 1e-6)
                examples_per_s = (batch_n * measured) / elapsed
                score = _score_train_batch_probe(
                    examples_per_s,
                    batch_n,
                    concurrent=concurrent,
                    target_positions_per_cycle=target_positions_per_cycle,
                )
                results.append({
                    "batch": batch_n,
                    "examples_per_s": round(examples_per_s, 3),
                    "elapsed_s": round(elapsed, 3),
                    "score": round(score, 4),
                })
            except RuntimeError as e:
                results.append({"batch": batch_n, "error": str(e)})
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            finally:
                restore()
                pbar.update(1)

    scored = [r for r in results if "examples_per_s" in r]
    if not scored:
        return {}, results
    best = max(scored, key=lambda r: (r["score"], r["examples_per_s"], -r["batch"]))
    return {"batch": best["batch"]}, results


def run_autotune_benchmark(cfg, backend, model, optimizer, device, hw, rust_binary, concurrent=True):
    overrides = {}
    benchmark = {}
    selfplay_meta = {}

    try:
        sp_overrides, sp_results, selfplay_meta = benchmark_selfplay_throughput(
            cfg, model, device, rust_binary, hw, concurrent=concurrent)
        overrides.update(sp_overrides)
        benchmark["selfplay"] = sp_results
    except Exception as e:
        benchmark["selfplay"] = [{"error": str(e)}]

    try:
        train_overrides, train_results = benchmark_train_batch(
            apply_runtime_overrides(cfg, overrides),
            backend,
            model,
            optimizer,
            device,
            hw,
            concurrent=concurrent,
            target_positions_per_cycle=selfplay_meta.get("positions_per_cycle"),
        )
        overrides.update(train_overrides)
        benchmark["train"] = train_results
    except Exception as e:
        benchmark["train"] = [{"error": str(e)}]

    if "bg_parallel" in overrides and "batch_size" not in overrides:
        overrides["batch_size"] = max(
            cfg.get("batch_size", 8),
            min(64, max(overrides["bg_parallel"] * overrides.get("n_threads", cfg.get("n_threads", 1)), 4)))
    return overrides, benchmark


def plan_online_runtime_overrides(cfg, hw, sample):
    overrides = {}
    parallel = max(1, cfg.get("bg_parallel", 1))
    batch_games = max(1, cfg.get("bg_batch_games", parallel))
    n_threads = max(1, cfg.get("n_threads", 1))
    batch = max(32, cfg.get("batch", 256))
    batch_multiple = 32 if batch >= 256 else 16

    last_cycle_s = float(sample.get("last_cycle_s", 0.0) or 0.0)
    last_cycle_positions = int(sample.get("last_cycle_positions", 0) or 0)
    rolling_cycle_s = float(sample.get("rolling_cycle_s", last_cycle_s) or last_cycle_s)
    positions_per_s = float(sample.get("rolling_positions_per_s", sample.get("positions_per_s", 0.0)) or 0.0)
    best_positions_per_s = float(sample.get("best_positions_per_s", positions_per_s) or positions_per_s)
    burst_ratio = float(sample.get("burst_ratio", 1.0) or 1.0)
    n_new = int(sample.get("n_new", 0) or 0)
    train_steps = int(sample.get("train_steps", 0) or 0)

    max_parallel = _autotune_parallel_limit(hw, concurrent=True)
    thread_capacity = _autotune_thread_capacity(hw, parallel)
    batch_game_cap = _autotune_batch_game_limit(hw, max_parallel, concurrent=True)

    if parallel > max_parallel:
        overrides["bg_parallel"] = max_parallel
        parallel = max_parallel
        thread_capacity = _autotune_thread_capacity(hw, parallel)
        batch_game_cap = _autotune_batch_game_limit(hw, max_parallel, concurrent=True)
        if batch_games > batch_game_cap:
            overrides["bg_batch_games"] = batch_game_cap

    if burst_ratio > 1.8 and batch_games > parallel:
        overrides["bg_batch_games"] = max(parallel, batch_games - parallel)
    elif rolling_cycle_s > 3.5 and batch_games > parallel:
        overrides["bg_batch_games"] = max(parallel, max(1, batch_games // 2))
    elif rolling_cycle_s < 1.5 and batch_games < batch_game_cap and positions_per_s >= best_positions_per_s * 0.95:
        overrides["bg_batch_games"] = min(batch_game_cap, batch_games + parallel)

    if (
        hw.device_kind != "cpu"
        and hw.gpu_vram_mb > 0
        and n_threads == 1
        and parallel >= max(4, max_parallel)
        and thread_capacity >= 2
    ):
        overrides["n_threads"] = min(thread_capacity, 2 if parallel >= 6 else 3)

    if hw.device_kind != "cpu" and hw.gpu_vram_mb > 0 and parallel <= 2 and n_threads > min(thread_capacity, 6):
        overrides["n_threads"] = min(thread_capacity, 6)

    if n_threads > thread_capacity and positions_per_s < best_positions_per_s * 0.95:
        overrides["n_threads"] = thread_capacity
    elif rolling_cycle_s < 2.0 and n_threads < thread_capacity and positions_per_s >= best_positions_per_s * 0.95:
        overrides["n_threads"] = min(thread_capacity, n_threads + 1)

    eff_parallel = overrides.get("bg_parallel", parallel)
    eff_threads = overrides.get("n_threads", n_threads)
    desired_batch_size = max(4, min(64, max(eff_parallel * eff_threads, 4)))
    if desired_batch_size != cfg.get("batch_size", 8):
        overrides["batch_size"] = desired_batch_size

    if max(last_cycle_positions, int(sample.get("rolling_positions", 0) or 0)) > 0:
        effective_positions = max(last_cycle_positions, int(sample.get("rolling_positions", 0) or 0))
        target_batch = _round_down_to_multiple(
            int(max(64, effective_positions * 3.5)), batch_multiple)
        target_batch = max(batch_multiple, target_batch)
        if n_new < batch * 0.5 and train_steps <= 3 and target_batch < batch:
            overrides["batch"] = target_batch
        elif n_new > batch * 1.5 and train_steps >= max(1, cfg.get("steps", 100) // 4) and target_batch > batch:
            overrides["batch"] = target_batch

    return overrides


class OnlineAutotuneController:
    def __init__(self, cfg, hw, enabled_iters=10, interval=2):
        self.cfg = cfg
        self.hw = hw
        self.enabled_iters = max(1, enabled_iters)
        self.interval = max(1, interval)
        self.best_positions_per_s = 0.0
        self._last_snapshot = None

    def observe(self, iteration_idx, n_new, elapsed_s, train_steps, replay_size, worker):
        if worker is None or iteration_idx >= self.enabled_iters:
            return {}
        if (iteration_idx + 1) % self.interval != 0:
            return {}

        snapshot = dict(worker.telemetry())
        self._last_snapshot = snapshot
        positions_per_s = float(n_new) / max(float(elapsed_s), 1e-6)
        self.best_positions_per_s = max(self.best_positions_per_s, positions_per_s)
        sample = {
            "iteration": iteration_idx + 1,
            "n_new": n_new,
            "elapsed_s": elapsed_s,
            "positions_per_s": positions_per_s,
            "best_positions_per_s": self.best_positions_per_s,
            "train_steps": train_steps,
            "replay_size": replay_size,
            "last_cycle_s": snapshot.get("last_cycle_s", 0.0),
            "last_cycle_positions": snapshot.get("last_cycle_positions", 0),
            "last_cycle_games": snapshot.get("last_cycle_games", 0),
            "rolling_cycle_s": snapshot.get("rolling_cycle_s", snapshot.get("last_cycle_s", 0.0)),
            "rolling_positions_per_s": snapshot.get("rolling_positions_per_s", positions_per_s),
            "rolling_positions": snapshot.get("rolling_positions", snapshot.get("last_cycle_positions", 0)),
            "rolling_games": snapshot.get("rolling_games", snapshot.get("last_cycle_games", 0)),
            "burst_ratio": snapshot.get("burst_ratio", 1.0),
        }
        overrides = plan_online_runtime_overrides(self.cfg, self.hw, sample)
        for key, value in list(overrides.items()):
            if self.cfg.get(key) == value:
                overrides.pop(key)
        if not overrides:
            return {}
        self.cfg.update(overrides)
        return overrides


def early_stopping_enabled(patience, concurrent=False):
    """Enable early stopping whenever patience is positive."""
    return patience > 0


def round_or_none(value, digits=4):
    return None if value is None else round(value, digits)


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def load_epoch_history(log_path):
    """Load per-epoch records from the JSONL training log."""
    history = []
    if not os.path.exists(log_path):
        return history
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("_type") == "eval":
                continue
            history.append(entry)
    return history


def build_elo_plot_series(history):
    """Build absolute-Elo and error-bar series for plotting."""
    eval_points = []
    for row in history:
        candidate_elo = row.get("published_elo")
        champion_elo = row.get("champion_elo")
        elo_gap = row.get("elo_gap")
        match_delta = row.get("delta_elo")
        if candidate_elo is None:
            continue
        if champion_elo is None and elo_gap is not None:
            champion_elo = candidate_elo - elo_gap
        elif champion_elo is None and match_delta is not None:
            champion_elo = candidate_elo - match_delta
        if elo_gap is None and champion_elo is not None:
            elo_gap = candidate_elo - champion_elo
        midpoint = None
        half_gap = None
        if champion_elo is not None and elo_gap is not None:
            midpoint = 0.5 * (candidate_elo + champion_elo)
            half_gap = abs(elo_gap) * 0.5
        eval_points.append({
            "iter": row.get("iter"),
            "candidate_elo": candidate_elo,
            "champion_elo": champion_elo,
            "elo_gap": elo_gap,
            "error_mid": midpoint,
            "error_half": half_gap,
            "score_rate": row.get("score_rate"),
            "match_delta_elo": match_delta,
            "eval_verdict": row.get("eval_verdict"),
        })
    return eval_points


def build_metric_plot_series(history, field):
    """Return sparse (iteration, value) pairs for a single plotted metric."""
    series = []
    for row in history:
        iteration = row.get("iter")
        value = row.get(field)
        if iteration is None or value is None:
            continue
        series.append((iteration, value))
    return series


def build_best_elo_series(elo_points):
    """Track champion Elo, promoting only on explicit promotion verdicts."""
    best_elo = []
    running_best = None
    for point in elo_points:
        champion_elo = point.get("champion_elo")
        candidate_elo = point.get("candidate_elo")
        verdict = point.get("eval_verdict")

        if champion_elo is not None:
            running_best = champion_elo if running_best is None else max(running_best, champion_elo)

        promoted = verdict == "promote"
        if verdict is None and champion_elo is not None and candidate_elo is not None:
            promoted = candidate_elo >= champion_elo and (point.get("match_delta_elo") or 0) > 0
        if promoted and candidate_elo is not None:
            running_best = candidate_elo if running_best is None else max(running_best, candidate_elo)

        best_elo.append(running_best)
    return best_elo


def generate_training_plots(log_path, output_dir):
    """Write training metric plots after training completes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] Plot generation skipped (matplotlib unavailable: {e})")
        return False

    history = load_epoch_history(log_path)
    if not history:
        return False

    iters = [h.get("iter") for h in history]
    loss = build_metric_plot_series(history, "loss")
    p_loss = build_metric_plot_series(history, "p_loss")
    v_loss = build_metric_plot_series(history, "v_loss")
    loss_ema = build_metric_plot_series(history, "loss_ema")
    elo_points = build_elo_plot_series(history)

    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_metric(series, label, **kwargs):
        if not series:
            return
        xs, ys = zip(*series)
        ax.plot(xs, ys, label=label, marker="o", markersize=3.5, **kwargs)

    plot_metric(loss, "loss", linewidth=2.0)
    plot_metric(p_loss, "p_loss", linewidth=1.5, alpha=0.9)
    plot_metric(v_loss, "v_loss", linewidth=1.5, alpha=0.9)
    if loss_ema:
        xs, ys = zip(*loss_ema)
        ax.plot(xs, ys, label="loss_ema", linewidth=2.0, linestyle="--", marker="o", markersize=3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.25)
    if iters:
        ax.set_xlim(min(iters), max(iters))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=140)
    plt.close(fig)

    if elo_points:
        from matplotlib.lines import Line2D

        elo_iters = [p["iter"] for p in elo_points]
        candidate_elo = [p["candidate_elo"] for p in elo_points]
        champion_elo = [p["champion_elo"] for p in elo_points]

        # --- Top panel: Elo progression ---
        fig, (ax_elo, ax_sr) = plt.subplots(2, 1, figsize=(10, 7),
            height_ratios=[3, 1], sharex=True)

        best_elo = build_best_elo_series(elo_points)

        # Best Elo: bold primary line
        ax_elo.plot(elo_iters, best_elo, color="#2563EB", linewidth=2.5,
                    marker="o", markersize=5, label="Best Elo", zorder=4)

        # Candidate & Champion: thin, muted
        ax_elo.plot(elo_iters, candidate_elo, color="#93C5FD", linewidth=1.0,
                    marker=".", markersize=3, label="Candidate", alpha=0.6, zorder=2)
        if any(v is not None for v in champion_elo):
            ax_elo.plot(elo_iters, champion_elo, color="#D1D5DB", linewidth=1.0,
                        linestyle="--", marker=".", markersize=3, label="Champion", alpha=0.5, zorder=1)

        # Error region: best Elo ± |delta_elo| from match measurement
        delta_data = [(it, be, p.get("match_delta_elo"))
                      for it, be, p in zip(elo_iters, best_elo, elo_points)
                      if be is not None and p.get("match_delta_elo") is not None]
        if delta_data:
            d_it, d_best, d_delta = zip(*delta_data)
            d_lo = [b - abs(d) for b, d in zip(d_best, d_delta)]
            d_hi = [b + abs(d) for b, d in zip(d_best, d_delta)]
            ax_elo.fill_between(d_it, d_lo, d_hi, alpha=0.12, color="#2563EB",
                                label="\u00b1 match \u0394Elo")

        ax_elo.set_ylabel("Elo Rating", fontsize=11)
        ax_elo.set_title("Elo Progression", fontsize=13, fontweight="bold")
        ax_elo.grid(True, alpha=0.2)
        ax_elo.legend(loc="upper left", fontsize=9, framealpha=0.9)
        # Auto-scale Y axis robustly (clip extreme outliers)
        all_elos = [v for v in candidate_elo + champion_elo if v is not None]
        if all_elos:
            q1 = sorted(all_elos)[len(all_elos) // 10]
            q9 = sorted(all_elos)[len(all_elos) * 9 // 10]
            iqr = max(q9 - q1, 100)
            ax_elo.set_ylim(min(all_elos[0], q1 - iqr * 0.3), q9 + iqr * 0.5)

        # --- Bottom panel: Score rate with promotion markers ---
        score_rate = [p.get("score_rate") for p in elo_points]
        if any(v is not None for v in score_rate):
            colors = []
            for sr in score_rate:
                if sr is None:
                    colors.append("#9CA3AF")
                elif sr > 0.55:
                    colors.append("#16A34A")  # green = promoted
                elif sr < 0.45:
                    colors.append("#DC2626")  # red = rejected
                else:
                    colors.append("#F59E0B")  # amber = marginal
            ax_sr.bar(elo_iters, [s if s is not None else 0 for s in score_rate],
                      color=colors, width=max(1, (max(elo_iters) - min(elo_iters)) / len(elo_iters) * 0.6),
                      edgecolor="none", alpha=0.85)
            ax_sr.axhline(y=0.5, color="#9CA3AF", linewidth=0.8, linestyle="--", alpha=0.6)
            ax_sr.axhline(y=0.55, color="#16A34A", linewidth=0.6, linestyle=":", alpha=0.4)
            ax_sr.set_ylabel("Score Rate", fontsize=10)
            ax_sr.set_ylim(0, 1)
            # Legend for bar colors
            ax_sr.legend(handles=[
                Line2D([0], [0], color="#16A34A", marker="s", linestyle="", markersize=7, label="Promoted (>55%)"),
                Line2D([0], [0], color="#F59E0B", marker="s", linestyle="", markersize=7, label="Marginal"),
                Line2D([0], [0], color="#DC2626", marker="s", linestyle="", markersize=7, label="Rejected (<45%)"),
            ], loc="upper right", fontsize=8, framealpha=0.9)

        ax_sr.set_xlabel("Iteration", fontsize=11)
        ax_sr.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_elo.png"), dpi=150)
        plt.close(fig)

    return True

# ═══════════════════════════════════════════
# § Early Stopping
# ═══════════════════════════════════════════

class EarlyStopping:
    """Stop training when smoothed loss stops improving."""
    def __init__(self, patience=10, min_delta=0.001, warmup=10, ema_alpha=0.3):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.ema_alpha = ema_alpha
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        self.num_updates = 0
        self.loss_ema = None

    def step(self, loss):
        self.num_updates += 1
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            a = self.ema_alpha
            self.loss_ema = a * loss + (1 - a) * self.loss_ema

        if self.num_updates <= self.warmup:
            if self.loss_ema < self.best_loss:
                self.best_loss = self.loss_ema
            return False

        if self.loss_ema < self.best_loss - self.min_delta:
            self.best_loss = self.loss_ema
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

# ═══════════════════════════════════════════
# § Eval Server (for Rust MCTS PythonIpcEval)
# ═══════════════════════════════════════════

def serve(model, cfg, device):
    model.eval()
    print(f"alphazero_server ready ({cfg['_name']})", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit": break
            features = req.get("features", [])
            action_mask = req.get("action_mask", [])
            n_act = req.get("num_actions", cfg['actions'])
            expected = cfg['ch'] * cfg['board'] * cfg['board']
            if len(features) == expected:
                x = torch.tensor(features, dtype=torch.float32).reshape(1, cfg['ch'], cfg['board'], cfg['board']).to(device)
            else:
                x = torch.zeros(1, cfg['ch'], cfg['board'], cfg['board'], device=device)
            with torch.no_grad():
                logits, val = model(x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            masked = np.zeros(n_act, dtype=np.float32)
            for i in range(min(len(action_mask), n_act)):
                if action_mask[i]: masked[i] = probs[i] if i < len(probs) else 0.0
            s = masked.sum()
            if s > 1e-8: masked /= s
            print(json.dumps({"status":"ok", "policy":masked.tolist(), "value":float(val.item())}), flush=True)
        except Exception as e:
            print(json.dumps({"status":"error", "policy":[], "value":0.0}), flush=True)

# ═══════════════════════════════════════════
# § Main
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# § Arena: Head-to-Head Model Comparison
# ═══════════════════════════════════════════

def arena_compare(model_a_path, model_b_path, cfg, device, n_games=50):
    """Play N games with SPRT early termination.
    
    SPRT: Sequential Probability Ratio Test
    H0: win_rate = 0.5 (equal strength)
    H1: win_rate = 0.55 (A is significantly stronger)
    α=β=0.05 → boundaries: lower=-2.944, upper=2.944
    """
    model_a = AlphaZeroNet(cfg).to(device)
    model_b = AlphaZeroNet(cfg).to(device)
    model_a.load_state_dict(load_torch_state_dict(model_a_path, torch, map_location=device))
    model_b.load_state_dict(load_torch_state_dict(model_b_path, torch, map_location=device))
    model_a.eval(); model_b.eval()

    wins_a, wins_b, draws = 0, 0, 0
    mcts_a = TreeMCTS(cfg, model_a, device)
    mcts_b = TreeMCTS(cfg, model_b, device)

    # SPRT parameters
    p0, p1 = 0.5, 0.55
    alpha, beta = 0.05, 0.05
    lower_bound = math.log(beta / (1 - alpha))      # ≈ -2.944
    upper_bound = math.log((1 - beta) / alpha)       # ≈ 2.944
    sprt_decided = False
    sprt_result = None

    with tqdm(total=n_games, desc="Arena", leave=False) as pbar:
        for game_idx in range(n_games):
            # Alternate colors
            if game_idx % 2 == 0:
                first, second = mcts_a, mcts_b
                first_is_a = True
            else:
                first, second = mcts_b, mcts_a
                first_is_a = False

            board = np.zeros(cfg['board']**2, dtype=np.int8)
            player = 1
            n2 = cfg['board']**2
            winner = 0

            for move_n in range(n2):
                legal_mask = np.array([1.0 if board[i]==0 else 0.0
                                       for i in range(min(n2, cfg['actions']))])
                if cfg['actions'] > n2:
                    legal_mask = np.concatenate([legal_mask, np.zeros(cfg['actions']-n2)])
                legal = [i for i in range(n2) if board[i]==0]
                if not legal: break

                encoded = np.zeros((cfg['ch'], cfg['board'], cfg['board']), dtype=np.float32)
                for i in range(n2):
                    r, c = i // cfg['board'], i % cfg['board']
                    if board[i] == player: encoded[0,r,c] = 1.0
                    elif board[i] != 0:   encoded[1,r,c] = 1.0
                if cfg['ch'] >= 3 and player == 1: encoded[2] = 1.0

                mcts = first if player == 1 else second
                policy = mcts.search(encoded, player, legal_mask, cfg['iters'] // 4)
                chosen = max(legal, key=lambda i: policy[i] if i < len(policy) else 0)
                board[chosen] = player

                # Check win
                if cfg['win'] > 0:
                    r0, c0 = chosen // cfg['board'], chosen % cfg['board']
                    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                        cnt = 1
                        for sign in [1,-1]:
                            nr, nc = r0+sign*dr, c0+sign*dc
                            while 0<=nr<cfg['board'] and 0<=nc<cfg['board'] and board[nr*cfg['board']+nc]==player:
                                cnt += 1; nr += sign*dr; nc += sign*dc
                        if cnt >= cfg['win']:
                            winner = player; break
                    if winner: break
                player = -player

            if winner == 1:
                if first_is_a: wins_a += 1
                else: wins_b += 1
            elif winner == -1:
                if first_is_a: wins_b += 1
                else: wins_a += 1
            else:
                draws += 1
            pbar.update(1)
            pbar.set_postfix_str(f"A:{wins_a} B:{wins_b} D:{draws}")

            # SPRT check after each decisive game
            decisive = wins_a + wins_b
            if decisive > 0 and not sprt_decided:
                w = wins_a
                n_dec = decisive
                # Log-likelihood ratio
                llr = w * math.log(p1/p0) + (n_dec - w) * math.log((1-p1)/(1-p0))
                if llr >= upper_bound:
                    sprt_decided = True
                    sprt_result = "H1_accept"  # A is stronger
                    pbar.set_postfix_str(f"SPRT: A wins (LLR={llr:.2f})")
                    break
                elif llr <= lower_bound:
                    sprt_decided = True
                    sprt_result = "H0_accept"  # no significant difference
                    pbar.set_postfix_str(f"SPRT: equal (LLR={llr:.2f})")
                    break

    total = wins_a + wins_b + draws
    wr = wins_a / max(total, 1)
    # Wilson score CI
    z = 1.96
    n = max(total, 1)
    p_hat = wr
    ci_lo = (p_hat + z*z/(2*n) - z*math.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n)) / (1+z*z/n)
    ci_hi = (p_hat + z*z/(2*n) + z*math.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n)) / (1+z*z/n)
    sprt_str = sprt_result or "inconclusive"
    return wins_a, wins_b, draws, wr, (ci_lo, ci_hi), sprt_str


# ═══════════════════════════════════════════
# § Rust NN-Backed Arena
# ═══════════════════════════════════════════

def _arena_rust_nn_impl(model_a_path, cfg_a, model_b_path, cfg_b, device, n_games=50,
                        rust_binary="./target/release/mcts_demo", strict=True):
    """Head-to-head using Rust MCTS with NN evaluation.

    Unlike arena_compare (Python TreeMCTS), this uses the full Rust search stack:
    TT, virtual loss, progressive widening, full QUARTZ controller.

    Args:
        strict: If True (default), raise error when Rust binary missing.
                If False, fall back to Python arena (NOT benchmark-grade).

    Each model gets its own NNSearchClient → separate Rust server process.
    """
    if cfg_a.get("_name") != cfg_b.get("_name"):
        raise ValueError(
            f"arena_rust_nn requires same game on both sides: {cfg_a.get('_name')} vs {cfg_b.get('_name')}"
        )

    is_chess = is_chess_game(cfg_a.get('_name'))

    model_a = AlphaZeroNet(cfg_a).to(device)
    model_b = AlphaZeroNet(cfg_b).to(device)
    model_a.load_state_dict(load_torch_state_dict(model_a_path, torch, map_location=device))
    model_b.load_state_dict(load_torch_state_dict(model_b_path, torch, map_location=device))
    model_a.eval(); model_b.eval()

    client_a = NNSearchClient(model_a, cfg_a, device, rust_binary)
    client_b = NNSearchClient(model_b, cfg_b, device, rust_binary)
    try:
        client_a.start(); client_b.start()
    except FileNotFoundError:
        if strict:
            raise RuntimeError(
                f"Arena (strict mode): Rust binary not found at {rust_binary}. "
                f"Run: cargo build --release. "
                f"Use strict=False for Python TreeMCTS fallback (NOT benchmark-grade).")
        print(f"  [WARN] Rust binary not found, falling back to Python arena (NOT benchmark-grade)")
        if cfg_a == cfg_b:
            return arena_compare(model_a_path, model_b_path, cfg_a, device, n_games)
        raise RuntimeError("strict=False fallback does not support asymmetric search configs")

    board_size = cfg_a['board']
    n2 = board_size ** 2
    n_actions = cfg_a['actions']
    win_len = cfg_a['win']
    penalty_mode_a = cfg_a.get('penalty_mode', 'GatedRefresh')
    penalty_mode_b = cfg_b.get('penalty_mode', 'GatedRefresh')

    wins_a, wins_b, draws = 0, 0, 0

    # SPRT parameters
    p0, p1 = 0.5, 0.55
    alpha, beta = 0.05, 0.05
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    sprt_decided = False
    sprt_result = None

    with tqdm(total=n_games, desc="Arena (Rust+NN)", leave=False) as pbar:
        for game_idx in range(n_games):
            if game_idx % 2 == 0:
                first_client, second_client = client_a, client_b
                first_is_a = True
            else:
                first_client, second_client = client_b, client_a
                first_is_a = False

            board = np.zeros(n2, dtype=np.int8) if not is_chess else None
            player = 1
            winner = 0
            current_fen = initial_chess_fen(cfg) if is_chess else None
            current_chess_meta = {} if is_chess else None
            max_moves = 500 if is_chess else n2

            for move_n in range(max_moves):
                client = first_client if player == 1 else second_client
                penalty_mode = (
                    penalty_mode_a if client is client_a else penalty_mode_b
                )

                if is_chess:
                    result = client.search_move(
                        None, player, penalty_mode, fen=current_fen, state_meta=current_chess_meta)
                else:
                    result = client.search_move(board, player, penalty_mode)
                if not result or 'error' in result:
                    break

                pol_entries = result.get('policy', [])
                if not pol_entries:
                    if is_chess:
                        terminal_value = float(result.get('value', 0.0))
                        if terminal_value < -0.5:
                            winner = -player
                        elif terminal_value > 0.5:
                            winner = player
                    break  # terminal

                if is_chess:
                    new_fen = result.get('result_fen', '')
                    if not new_fen or new_fen == current_fen:
                        break
                    current_fen = new_fen
                    current_chess_meta = chess_state_meta_from_hashes(
                        result.get("result_history_hashes", []))
                    player = -player
                else:
                    best = result.get('best_move', -1)
                    legal = [i for i in range(n2) if board[i] == 0]
                    if best < 0 or best >= n2 or board[best] != 0:
                        if legal: best = random.choice(legal)
                        else: break

                    board[best] = player

                    if win_len > 0:
                        r0, c0 = best // board_size, best % board_size
                        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                            cnt = 1
                            for sign in [1,-1]:
                                nr, nc = r0+sign*dr, c0+sign*dc
                                while 0<=nr<board_size and 0<=nc<board_size and board[nr*board_size+nc]==player:
                                    cnt += 1; nr += sign*dr; nc += sign*dc
                            if cnt >= win_len: winner = player; break
                        if winner: break

                    if not [i for i in range(n2) if board[i] == 0]:
                        break
                    player = -player

            if winner == 1:
                if first_is_a: wins_a += 1
                else: wins_b += 1
            elif winner == -1:
                if first_is_a: wins_b += 1
                else: wins_a += 1
            else:
                draws += 1

            pbar.update(1)
            pbar.set_postfix_str(f"A:{wins_a} B:{wins_b} D:{draws}")

            # SPRT
            decisive = wins_a + wins_b
            if decisive > 0 and not sprt_decided:
                llr = wins_a * math.log(p1/p0) + (decisive - wins_a) * math.log((1-p1)/(1-p0))
                if llr >= upper_bound:
                    sprt_decided = True; sprt_result = "H1_accept"; break
                elif llr <= lower_bound:
                    sprt_decided = True; sprt_result = "H0_accept"; break

    client_a.stop(); client_b.stop()

    total = wins_a + wins_b + draws
    wr = wins_a / max(total, 1)
    z = 1.96; n = max(total, 1); p_hat = wr
    ci_lo = (p_hat + z*z/(2*n) - z*math.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n)) / (1+z*z/n)
    ci_hi = (p_hat + z*z/(2*n) + z*math.sqrt((p_hat*(1-p_hat)+z*z/(4*n))/n)) / (1+z*z/n)
    sprt_str = sprt_result or "inconclusive"
    return wins_a, wins_b, draws, wr, (ci_lo, ci_hi), sprt_str


def arena_rust_nn(model_a_path, model_b_path, cfg, device, n_games=50,
                  rust_binary="./target/release/mcts_demo", strict=True):
    return _arena_rust_nn_impl(
        model_a_path,
        cfg,
        model_b_path,
        cfg,
        device,
        n_games=n_games,
        rust_binary=rust_binary,
        strict=strict,
    )


def arena_rust_nn_dual_cfg(model_a_path, cfg_a, model_b_path, cfg_b, device, n_games=50,
                           rust_binary="./target/release/mcts_demo", strict=True):
    return _arena_rust_nn_impl(
        model_a_path,
        cfg_a,
        model_b_path,
        cfg_b,
        device,
        n_games=n_games,
        rust_binary=rust_binary,
        strict=strict,
    )


# ═══════════════════════════════════════════
# § Glicko-2 Rating System + 3-Agent Arena
# ═══════════════════════════════════════════

class Glicko2Rating:
    """Glicko-2 rating for a single player."""
    def __init__(self, mu=1500.0, phi=350.0, sigma=0.06):
        self.mu = mu        # rating
        self.phi = phi      # rating deviation (uncertainty)
        self.sigma = sigma  # volatility

    def to_dict(self):
        return {"mu": self.mu, "phi": self.phi, "sigma": self.sigma}

    @staticmethod
    def from_dict(d):
        return Glicko2Rating(d["mu"], d["phi"], d["sigma"])


class Glicko2System:
    """Glicko-2 rating system with deflation protection.
    
    Deflation protection: anchor agent (random rollout) is pinned at 1000.
    After each rating period, all ratings are shifted so anchor stays at 1000.
    This prevents rating deflation as models improve.
    """
    TAU = 0.5  # system volatility constant

    def __init__(self, path=None):
        self.ratings = {}  # name → Glicko2Rating
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    def ensure(self, name, mu=1500.0, phi=350.0):
        if name not in self.ratings:
            self.ratings[name] = Glicko2Rating(mu, phi)
        return self.ratings[name]

    def _g(self, phi):
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / (math.pi**2))

    def _E(self, mu, muj, phij):
        return 1.0 / (1.0 + math.exp(-self._g(phij) * (mu - muj)))

    def update(self, name, opponents_results):
        """Update rating after a rating period.
        
        opponents_results: list of (opponent_name, score) where score in {0, 0.5, 1}
        """
        r = self.ensure(name)
        if not opponents_results:
            # No games: increase uncertainty
            r.phi = min(350.0, math.sqrt(r.phi**2 + r.sigma**2))
            return

        # Convert to Glicko-2 scale
        mu = (r.mu - 1500.0) / 173.7178
        phi = r.phi / 173.7178

        # Compute v (estimated variance)
        v_inv = 0.0
        delta_sum = 0.0
        for opp_name, score in opponents_results:
            opp = self.ensure(opp_name)
            muj = (opp.mu - 1500.0) / 173.7178
            phij = opp.phi / 173.7178
            g_val = self._g(phij)
            E_val = self._E(mu, muj, phij)
            v_inv += g_val**2 * E_val * (1 - E_val)
            delta_sum += g_val * (score - E_val)

        if v_inv < 1e-12:
            return
        v = 1.0 / v_inv
        delta = v * delta_sum

        # Update volatility (simplified Illinois algorithm)
        a = math.log(r.sigma**2)
        tau2 = self.TAU**2
        phi2 = phi**2

        def f(x):
            ex = math.exp(x)
            d2 = delta**2
            num1 = ex * (d2 - phi2 - v - ex)
            den1 = 2.0 * (phi2 + v + ex)**2
            return num1 / den1 - (x - a) / tau2

        # Bisection
        A = a
        if delta**2 > phi2 + v:
            B = math.log(delta**2 - phi2 - v)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
                if k > 100: break
            B = a - k * self.TAU

        for _ in range(50):
            C = (A + B) / 2.0
            if abs(B - A) < 1e-6:
                break
            if f(C) * f(A) < 0:
                B = C
            else:
                A = C

        sigma_new = math.exp(C / 2.0)
        phi_star = math.sqrt(phi2 + sigma_new**2)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        mu_new = mu + phi_new**2 * delta_sum

        r.mu = mu_new * 173.7178 + 1500.0
        r.phi = phi_new * 173.7178
        r.sigma = sigma_new

    def deflation_adjust(self, anchor_name="random_rollout", anchor_target=1000.0):
        """Pin anchor agent at target rating → prevents deflation."""
        if anchor_name not in self.ratings:
            return
        drift = self.ratings[anchor_name].mu - anchor_target
        if abs(drift) > 1.0:
            for r in self.ratings.values():
                r.mu -= drift

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1].mu)

    def save(self, path=None):
        p = path or self.path
        if p:
            with open(p, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.ratings.items()}, f, indent=2)

    def load(self, path=None):
        p = path or self.path
        if p and os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            self.ratings = {k: Glicko2Rating.from_dict(v) for k, v in data.items()}


class RandomRolloutAgent:
    """Anchor agent: plays random legal moves. Fixed strength baseline."""
    def choose_move(self, board, player, board_size):
        n2 = board_size ** 2
        legal = [i for i in range(n2) if board[i] == 0]
        return random.choice(legal) if legal else -1


def arena_3agent(model_current_path, model_best_path, cfg, device,
                 games_per_pair=20, rust_binary="./target/release/mcts_demo",
                 use_rust_nn=False, rating_path=None):
    """3-agent round-robin arena with Glicko-2 ratings.
    
    Agents:
      1. random_rollout — anchor (pinned at 1000 Glicko)
      2. current — model being trained
      3. best — highest-rated checkpoint so far
    
    Each pair plays games_per_pair games (alternating colors).
    After all games, Glicko-2 ratings are updated with deflation correction.
    
    Returns: (ratings_dict, current_promoted: bool)
    """
    glicko = Glicko2System(rating_path)
    glicko.ensure("random_rollout", mu=1000.0, phi=100.0)
    glicko.ensure("current", mu=1500.0, phi=200.0)
    glicko.ensure("best", mu=1500.0, phi=200.0)

    board_size = cfg['board']
    n2 = board_size ** 2
    win_len = cfg['win']
    rand_agent = RandomRolloutAgent()

    def play_game(agent_a_fn, agent_b_fn, swap=False):
        """Play one game. Returns (score_a, score_b)."""
        board = np.zeros(n2, dtype=np.int8)
        player = 1
        for move_n in range(n2):
            fn = agent_a_fn if (player == 1) != swap else agent_b_fn
            move = fn(board, player)
            if move < 0 or move >= n2 or board[move] != 0:
                return (0.0, 1.0) if ((player == 1) != swap) else (1.0, 0.0)
            board[move] = player
            if win_len > 0:
                r0, c0 = move // board_size, move % board_size
                for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
                    cnt = 1
                    for sign in [1,-1]:
                        nr, nc = r0+sign*dr, c0+sign*dc
                        while 0<=nr<board_size and 0<=nc<board_size and board[nr*board_size+nc]==player:
                            cnt += 1; nr += sign*dr; nc += sign*dc
                    if cnt >= win_len:
                        if (player == 1) != swap: return (1.0, 0.0)
                        else: return (0.0, 1.0)
            if not [i for i in range(n2) if board[i] == 0]:
                return (0.5, 0.5)
            player = -player
        return (0.5, 0.5)

    # Load models
    model_curr = AlphaZeroNet(cfg).to(device)
    model_curr.load_state_dict(load_torch_state_dict(model_current_path, torch, map_location=device))
    model_curr.eval()

    if model_best_path and os.path.exists(model_best_path):
        model_best = AlphaZeroNet(cfg).to(device)
        model_best.load_state_dict(load_torch_state_dict(model_best_path, torch, map_location=device))
        model_best.eval()
    else:
        model_best = model_curr

    # Create search functions
    mcts_curr = TreeMCTS(cfg, model_curr, device)
    mcts_best = TreeMCTS(cfg, model_best, device)
    iters = cfg['iters'] // 4

    def nn_move(mcts, board, player):
        enc = encode_board(cfg, np.array(board, dtype=np.int8) if not isinstance(board, np.ndarray) else board, player)
        legal_mask = np.array([1.0 if board[i]==0 else 0.0 for i in range(min(n2, cfg['actions']))])
        if cfg['actions'] > n2: legal_mask = np.concatenate([legal_mask, np.zeros(cfg['actions']-n2)])
        pol = mcts.search(enc, player, legal_mask, iters)
        legal = [i for i in range(n2) if board[i] == 0]
        if not legal: return -1
        return max(legal, key=lambda a: pol[a] if a < len(pol) else 0)

    curr_fn = lambda b, p: nn_move(mcts_curr, b, p)
    best_fn = lambda b, p: nn_move(mcts_best, b, p)
    rand_fn = lambda b, p: rand_agent.choose_move(b, p, board_size)

    # Round-robin: 3 pairs
    pairs = [
        ("current", "random_rollout", curr_fn, rand_fn),
        ("best", "random_rollout", best_fn, rand_fn),
        ("current", "best", curr_fn, best_fn),
    ]

    results = {name: [] for name in ["current", "best", "random_rollout"]}

    print("  3-Agent Round-Robin Arena:")
    for name_a, name_b, fn_a, fn_b in pairs:
        wa, wb, d = 0, 0, 0
        for gi in range(games_per_pair):
            swap = gi % 2 == 1
            sa, sb = play_game(fn_a, fn_b, swap)
            if sa > sb: wa += 1
            elif sb > sa: wb += 1
            else: d += 1
            results[name_a].append((name_b, sa))
            results[name_b].append((name_a, sb))
        print(f"    {name_a} vs {name_b}: {wa}-{wb}-{d}")

    # Glicko-2 update
    for name, opp_results in results.items():
        glicko.update(name, opp_results)

    # Deflation correction: pin random_rollout at 1000
    glicko.deflation_adjust("random_rollout", 1000.0)
    glicko.save()

    print("  Ratings (Glicko-2, deflation-adjusted):")
    for name, r in glicko.leaderboard():
        print(f"    {name:20s}  {r.mu:7.1f} ± {r.phi:.1f}")

    # Promotion: current > best?
    curr_r = glicko.ratings.get("current", Glicko2Rating())
    best_r = glicko.ratings.get("best", Glicko2Rating())
    promoted = curr_r.mu > best_r.mu + 30  # require 30-point margin
    if promoted:
        print(f"  → PROMOTED: current ({curr_r.mu:.0f}) > best ({best_r.mu:.0f})")
    else:
        print(f"  → NOT promoted: current ({curr_r.mu:.0f}) vs best ({best_r.mu:.0f})")

    return {k: v.to_dict() for k, v in glicko.ratings.items()}, promoted


# ═══════════════════════════════════════════
# § Batched Rust NN Self-Play (N-game parallel)
# ═══════════════════════════════════════════

def selfplay_rust_nn_batched(cfg, model, device, n_games, rust_binary="./target/release/mcts_demo",
                              parallel=4, show_progress=True, proc_pool=None, perf_stats=None,
                              on_game=None, active_proc_ref=None):
    """Run N games in parallel via a single Rust server + shared batched NN eval."""
    board_size = cfg['board']
    n_actions = cfg['actions']
    penalty_mode = cfg.get('penalty_mode', 'GatedRefresh')
    iters = cfg['iters']
    is_chess = is_chess_game(cfg.get('_name'))
    use_resident_session = (
        not bool(cfg.get("_disable_resident_session", False))
        and should_use_resident_session(
            cfg.get('_name'),
            parallel,
            n_games,
            enabled=bool(cfg.get("_resident_session", False)),
        )
    )
    rust_game = rust_game_name(cfg['_name'])
    max_moves = 500 if is_chess or is_go_game(cfg.get('_name')) else (board_size ** 2 + 5)

    all_states, all_policies, all_outcomes, all_traces = [], [], [], []
    games_done = 0
    if model is not None and hasattr(model, "eval"):
        model.eval()
    proc = None

    if (
        cfg.get("_selfplay_runner_mode") == "rust_selfplay_state_machine"
        and supports_rust_selfplay_state_machine(cfg.get('_name'))
        and not is_chess
    ):
        client = NNSearchClient(model, cfg, device, rust_binary)
        try:
            client.start()
            if active_proc_ref is not None:
                active_proc_ref._active_proc = client.proc
            def _handle_stream_chunk(games):
                for game_payload in games:
                    states, policies, outcome, traces = _decode_streamed_selfplay_game(cfg, game_payload)
                    all_states.append(states)
                    all_policies.append(policies)
                    all_outcomes.append(outcome)
                    all_traces.append(traces)
                    if callable(on_game):
                        on_game(states, policies, outcome, traces)

            try:
                payload = client.selfplay_run(
                    n_games=n_games,
                    parallel=parallel,
                    temp_threshold=cfg['temp_th'],
                    penalty_mode=penalty_mode,
                    seed=random.randint(0, 2**31),
                    on_chunk=_handle_stream_chunk,
                )
            except TypeError:
                payload = client.selfplay_run(
                    n_games=n_games,
                    parallel=parallel,
                    temp_threshold=cfg['temp_th'],
                    penalty_mode=penalty_mode,
                    seed=random.randint(0, 2**31),
                )
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload.get("error")))
            games = payload.get("games", []) if isinstance(payload, dict) else []
            if isinstance(games, list) and games:
                _handle_stream_chunk(games)
            if isinstance(payload, dict):
                completed_games = int(payload.get("completed_games", len(all_states)) or 0)
                if completed_games == len(all_states) == n_games:
                    return all_states, all_policies, all_outcomes, all_traces
        finally:
            client.stop()

    if perf_stats is not None:
        perf_stats.setdefault("eval_messages", 0)
        perf_stats.setdefault("eval_items", 0)
        perf_stats.setdefault("model_calls", 0)
        perf_stats.setdefault("model_batch_sizes", [])
        perf_stats.setdefault("model_time_s", 0.0)
        perf_stats.setdefault("collect_loops", 0)
        perf_stats.setdefault("result_messages", 0)

    def build_game_data():
        gd = {
            'player': 1,
            'moves': 0,
            'states': [], 'policies': [],
            'finished': False, 'winner': 0.0,
            'void_result': False,
            'trace': [],
        }
        if is_chess:
            gd['fen'] = initial_chess_fen(cfg)
            gd['chess_history_hashes'] = []
        else:
            gd['state'] = build_training_game_adapter(cfg)
        return gd

    def build_job(gd):
        if is_chess:
            job = {'fen': gd['fen']}
            job.update(chess_state_meta_from_hashes(gd.get('chess_history_hashes', [])))
            return job
        state = gd['state']
        player = 1 if state.current_player() == 0 else -1
        job = {
            'player': player,
            'board': normalize_rust_board(rust_game, state._board),
        }
        job.update(build_rust_state_meta(cfg.get('_name'), state, cfg))
        return job

    def apply_result(gd, result):
        pol_entries = result.get('policy', [])
        if not pol_entries:
            if is_chess:
                gd['winner'] = result.get('value', 0.0) * gd['player']
            else:
                state = gd['state']
                if hasattr(state, 'is_void_result') and state.is_void_result():
                    gd['void_result'] = True
                    gd['winner'] = 0.0
                else:
                    gd['winner'] = float(state.outcome_for_black() or 0.0)
            gd['finished'] = True
            return None

        policy = dense_policy_from_sparse(pol_entries, n_actions)

        enc = encode_chess_fen(gd['fen']) if is_chess else gd['state']._encode()
        gd['states'].append(enc.copy())
        gd['policies'].append(policy.copy())
        gd['trace'].append({
            'p_flip': result.get('p_flip', 0.0),
            'value': result.get('value', 0.0),
            'sigma_q': result.get('sigma_q', 0.0),
            'stop_reason': result.get('stop_reason', ''),
            'hbar_eff': result.get('hbar_eff', 0.0),
            'iterations': result.get('iterations', 0),
            'dup_rate': result.get('dup_rate', 0.0),
            'max_pending': result.get('max_pending', 0),
            'avg_vvalue': result.get('avg_vvalue', 0.0),
        })

        if is_chess:
            new_fen = result.get('result_fen', '')
            if not new_fen or new_fen == gd['fen']:
                gd['finished'] = True
                gd['winner'] = result.get('value', 0.0) * gd['player']
            else:
                gd['fen'] = new_fen
                gd['chess_history_hashes'] = [
                    int(v) for v in result.get("result_history_hashes", [])
                ]
                gd['moves'] += 1
                gd['player'] = -gd['player']
            return None

        state = gd['state']
        legal = state.legal_moves()
        chosen = choose_selfplay_move(
            policy, legal, gd['moves'], cfg['temp_th'],
            fallback_best=result.get('best_move', -1))
        state.apply_move(chosen)
        gd['moves'] += 1
        if state.is_terminal():
            gd['finished'] = True
            if hasattr(state, 'is_void_result') and state.is_void_result():
                gd['void_result'] = True
                gd['winner'] = 0.0
            else:
                gd['winner'] = float(state.outcome_for_black() or 0.0)
            return None
        return chosen

    def parse_eval_group(kind, payload):
        if kind == "frame":
            frame_kind, frame_payload = proc_decode_eval_frame(proc, payload[0], payload[1])
            if frame_kind == QIPC_BATCH_EVAL_REQ:
                requests = unpack_qipc_batch_eval_req(frame_payload)
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += len(requests)
                return _make_eval_request_group(
                    "binary_batch",
                    requests,
                    gi=0,
                    prefer_shm=True,
                ), None
            if frame_kind == QIPC_EVAL_REQ:
                na, feats, model_tag = unpack_qipc_eval_req(frame_payload)
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += 1
                return _make_eval_request_group(
                    "binary_single",
                    [(na, feats, model_tag)],
                    gi=0,
                    prefer_shm=True,
                ), None
            return None, {"error": f"unexpected frame kind {frame_kind}"}
        if kind == "json" and isinstance(payload, dict):
            if "batch_eval_req" in payload:
                parsed = payload["batch_eval_req"]
                reqs = [
                    (int(r.get('num_actions', n_actions)), r.get('features', []))
                    for r in parsed.get("requests", [])
                ]
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += len(reqs)
                return _make_eval_request_group("json_batch", reqs, gi=0), None
            if "eval_req" in payload:
                er = payload["eval_req"]
                if perf_stats is not None:
                    perf_stats["eval_messages"] += 1
                    perf_stats["eval_items"] += 1
                return _make_eval_request_group(
                    "json_single",
                    [(int(er.get('num_actions', n_actions)), er.get('features', []))],
                    gi=0,
                ), None
            if "results" in payload:
                return None, payload
            if "error" in payload:
                return None, payload
        return None, {"error": "unexpected message"}

    _duty = {"read_s": 0.0, "collect_s": 0.0, "model_s": 0.0, "write_s": 0.0, "cycles": 0}
    _duty_log_interval = 16
    _use_pipeline = (
        not os.environ.get("QUARTZ_DISABLE_ASYNC_PIPELINE")
        and model is not None
        and not hasattr(model, "predict")  # torch only — JAX has no GIL-release guarantee
    )

    def exchange_search_request(req):
        nonlocal batch_items_ema, collect_wait_ema_s
        req_cmd = req.get("cmd", "?") if isinstance(req, dict) else "?"
        req_jobs = len(req.get("jobs", [])) if isinstance(req, dict) and isinstance(req.get("jobs"), list) else None
        req_updates = len(req.get("updates", [])) if isinstance(req, dict) and isinstance(req.get("updates"), list) else None
        req_t0 = time.perf_counter()
        _stall_trace(
            "exchange_begin",
            cmd=req_cmd,
            jobs=req_jobs,
            updates=req_updates,
            parallel=int(parallel),
            n_games=int(n_games),
            resident=bool(use_resident_session),
        )
        proc_write_json_line(proc, req)

        # --- SHM ring buffer fast path ---
        _ring = getattr(proc, "_quartz_ring_buffer", None)
        if _ring is not None:
            ring_payload = _shm_eval_loop(_ring, model, device, cfg, proc)
            _stall_trace("exchange_end", cmd=req_cmd, loops=0,
                         elapsed_s=float(time.perf_counter() - req_t0))
            if perf_stats is not None and ring_payload is not None:
                perf_stats["result_messages"] += 1
            if isinstance(ring_payload, dict):
                return ring_payload
            kind, payload = proc_read_message(proc)
            if kind == "json" and isinstance(payload, dict):
                return payload
            return None

        deferred = None
        results_payload = None
        loop_count = 0

        # Pipeline state: at most one batch in-flight on inference thread
        pipeline = None
        inflight = False
        pending_response = None  # completed inference results waiting to be written

        if _use_pipeline:
            pipeline = InferencePipelineThread(model, device, cfg, max_pending=1)
            pipeline.start()

        try:
          while results_payload is None:
            loop_count += 1
            if perf_stats is not None:
                perf_stats["collect_loops"] += 1

            # --- Flush: wait for inflight inference and write result ---
            # MUST happen before read: Rust broker won't send the next request
            # until it receives the response for the current one.
            if inflight and pipeline is not None:
                model_t0 = time.perf_counter()
                flush_responses = pipeline.collect(timeout=30.0)
                _duty["model_s"] += time.perf_counter() - model_t0
                inflight = False
                write_t0 = time.perf_counter()
                for rg in flush_responses:
                    _write_batched_eval_group(proc, rg)
                _duty["write_s"] += time.perf_counter() - write_t0

            # --- Read first request ---
            read_t0 = time.perf_counter()
            kind, payload = deferred if deferred is not None else proc_read_message(proc)
            _duty["read_s"] += time.perf_counter() - read_t0
            _stall_trace(
                "exchange_message",
                cmd=req_cmd,
                loop=int(loop_count),
                kind=kind,
                read_wait_s=float(time.perf_counter() - read_t0),
                deferred=bool(deferred is not None),
            )
            deferred = None
            if kind is None:
                _stall_trace("exchange_eof", cmd=req_cmd, loop=int(loop_count))
                return None

            first_group, terminal = parse_eval_group(kind, payload)
            if terminal is not None:
                results_payload = terminal
                _stall_trace(
                    "exchange_terminal",
                    cmd=req_cmd,
                    loop=int(loop_count),
                    elapsed_s=float(time.perf_counter() - req_t0),
                    keys=sorted(list(terminal.keys())) if isinstance(terminal, dict) else None,
                )
                break

            # --- Collect batch ---
            eval_groups = [first_group]
            eval_item_count = len(first_group["requests"])
            dynamic_target_eval_items, dynamic_collect_timeout_s = compute_eval_collect_policy(
                base_target_eval_items,
                base_collect_timeout_s,
                batch_items_ema=batch_items_ema,
                wait_ema_s=collect_wait_ema_s,
            )
            collect_t0 = time.perf_counter()
            deadline = time.perf_counter() + dynamic_collect_timeout_s
            while eval_item_count < dynamic_target_eval_items:
                timeout_s = max(0.0, deadline - time.perf_counter())
                if timeout_s <= 0.0:
                    break
                if not wait_readable(proc.stdout, timeout_s):
                    break
                next_kind, next_payload = proc_read_message(proc)
                next_group, next_terminal = parse_eval_group(next_kind, next_payload)
                if next_terminal is not None:
                    deferred = (next_kind, next_payload)
                    break
                eval_groups.append(next_group)
                eval_item_count += len(next_group["requests"])

            merged_items = sum(len(group["requests"]) for group in eval_groups)
            collect_wait_s = time.perf_counter() - collect_t0
            _duty["collect_s"] += collect_wait_s
            batch_items_ema = 0.25 * float(merged_items) + 0.75 * float(batch_items_ema)
            collect_wait_ema_s = 0.25 * float(collect_wait_s) + 0.75 * float(collect_wait_ema_s)

            # --- Inference: submit to pipeline or run sync ---
            if pipeline is not None:
                pipeline.submit(eval_groups)
                inflight = True
                _stall_trace(
                    "exchange_eval",
                    cmd=req_cmd,
                    loop=int(loop_count),
                    groups=int(len(eval_groups)),
                    items=int(merged_items),
                    target_items=int(dynamic_target_eval_items),
                    collect_wait_s=float(collect_wait_s),
                    collect_timeout_s=float(dynamic_collect_timeout_s),
                    eval_s=0.0,
                    pipelined=True,
                )
            else:
                t_eval = time.perf_counter()
                responses = _run_batched_eval_groups(eval_groups, model, device, cfg)
                eval_elapsed = time.perf_counter() - t_eval
                _duty["model_s"] += eval_elapsed
                _stall_trace(
                    "exchange_eval",
                    cmd=req_cmd,
                    loop=int(loop_count),
                    groups=int(len(eval_groups)),
                    items=int(merged_items),
                    target_items=int(dynamic_target_eval_items),
                    collect_wait_s=float(collect_wait_s),
                    collect_timeout_s=float(dynamic_collect_timeout_s),
                    eval_s=float(eval_elapsed),
                )
                write_t0 = time.perf_counter()
                for response_group in responses:
                    _write_batched_eval_group(proc, response_group)
                _duty["write_s"] += time.perf_counter() - write_t0

            if perf_stats is not None:
                perf_stats["model_calls"] += 1
                perf_stats["model_batch_sizes"].append(merged_items)
            _duty["cycles"] += 1
            if _duty["cycles"] % _duty_log_interval == 0:
                NNSearchClient._emit_duty_cycle(_duty)
        finally:
            if inflight and pipeline is not None:
                try:
                    drain = pipeline.collect(timeout=10.0)
                    for rg in drain:
                        _write_batched_eval_group(proc, rg)
                except Exception:
                    pass
            if pipeline is not None:
                pipeline.stop()

        if _duty["cycles"] > 0:
            NNSearchClient._emit_duty_cycle(_duty)
        if perf_stats is not None and results_payload is not None:
            perf_stats["result_messages"] += 1
        _stall_trace(
            "exchange_end",
            cmd=req_cmd,
            loops=int(loop_count),
            elapsed_s=float(time.perf_counter() - req_t0),
        )
        return results_payload

    proc = proc_pool.acquire(1)[0] if proc_pool is not None else launch_rust_server(rust_binary)
    if active_proc_ref is not None:
        active_proc_ref._active_proc = proc
    try:
        slot_count = min(max(1, parallel), max(1, n_games))
        with tqdm(total=n_games, desc="Self-play (Rust+NN batched)", leave=False,
                  disable=not show_progress) as pbar:
            game_data = [None] * slot_count
            games_started = 0

            def launch_slot(gi):
                nonlocal games_started
                if games_started >= n_games:
                    game_data[gi] = None
                    return False
                game_data[gi] = build_game_data()
                games_started += 1
                return True

            def finalize_slot(gi):
                nonlocal games_done
                gd = game_data[gi]
                if gd is None:
                    return
                if gd.get('void_result', False):
                    all_states.append([])
                    all_policies.append([])
                else:
                    all_states.append(gd['states'])
                    all_policies.append(gd['policies'])
                all_outcomes.append(float(gd.get('winner', 0.0)))
                all_traces.append(gd.get('trace', []))
                pbar.update(1)
                games_done += 1
                if not launch_slot(gi):
                    game_data[gi] = None

            for gi in range(slot_count):
                launch_slot(gi)

            search_opts = rust_search_options(cfg, penalty_mode=penalty_mode)
            base_collect_timeout_s = min(
                0.006,
                max(0.00075, float(search_opts.get("batch_timeout_us", 1500)) / 1_000_000.0 * 0.9),
            )
            base_target_eval_items = max(1, int(search_opts.get("batch_size", cfg.get("batch_size", 8))))
            batch_items_ema = float(base_target_eval_items)
            collect_wait_ema_s = 0.0
            if use_resident_session:
                session_req = {
                    'cmd': 'search_nn_multi_session_open',
                    'game': rust_game,
                    'iters': iters,
                    'jobs': [build_job(gd) for gd in game_data if gd is not None],
                }
                session_req.update(search_opts)
                results_payload = exchange_search_request(session_req)
                session_id = results_payload.get("session_id") if isinstance(results_payload, dict) else None
                results = results_payload.get("results", []) if isinstance(results_payload, dict) else []

                while games_done < n_games:
                    if not isinstance(results, list) or len(results) != slot_count:
                        for gi, gd in enumerate(game_data):
                            if gd is not None:
                                gd['finished'] = True
                                gd['winner'] = 0.0
                                finalize_slot(gi)
                        break

                    updates = []
                    for gi, result in enumerate(results):
                        gd = game_data[gi]
                        if gd is None:
                            updates.append({"deactivate": True})
                            continue
                        chosen = apply_result(gd, result if isinstance(result, dict) else {})
                        if gd['moves'] >= max_moves and not gd['finished']:
                            gd['finished'] = True
                            gd['winner'] = 0.0
                        if gd['finished']:
                            finalize_slot(gi)
                            gd = game_data[gi]
                            if gd is None:
                                updates.append({"deactivate": True})
                            else:
                                updates.append({"replace": build_job(gd)})
                        else:
                            updates.append({"action": int(chosen)} if chosen is not None else {})

                    if games_done >= n_games and all(gd is None for gd in game_data):
                        break
                    if session_id is None:
                        break
                    results_payload = exchange_search_request({
                        "cmd": "search_nn_multi_session_step",
                        "session_id": int(session_id),
                        "updates": updates,
                    })
                    results = results_payload.get("results", []) if isinstance(results_payload, dict) else []

                if session_id is not None:
                    try:
                        proc_write_json_line(proc, {"cmd": "search_nn_multi_session_close", "session_id": int(session_id)})
                        proc_read_json_line(proc)
                    except Exception:
                        pass
            else:
                while games_done < n_games:
                    active = []
                    jobs = []
                    for gi, gd in enumerate(game_data):
                        if gd is None:
                            continue
                        if gd['finished'] or gd['moves'] >= max_moves:
                            if gd['moves'] >= max_moves and not gd['finished']:
                                gd['finished'] = True
                                gd['winner'] = 0.0
                            finalize_slot(gi)
                            gd = game_data[gi]
                        if gd is None:
                            continue
                        active.append(gi)
                        jobs.append(build_job(gd))

                    if not active:
                        continue

                    req = {
                        'cmd': 'search_nn_multi',
                        'game': rust_game,
                        'iters': iters,
                        'jobs': jobs,
                    }
                    req.update(search_opts)
                    results_payload = exchange_search_request(req)
                    results = []
                    if isinstance(results_payload, dict):
                        results = results_payload.get("results", [])
                    if not isinstance(results, list) or len(results) != len(active):
                        for gi in active:
                            gd = game_data[gi]
                            if gd is not None:
                                gd['finished'] = True
                                gd['winner'] = 0.0
                        continue
                    for gi, result in zip(active, results):
                        gd = game_data[gi]
                        if gd is not None:
                            apply_result(gd, result if isinstance(result, dict) else {})
    finally:
        if proc_pool is None:
            stop_rust_server(proc)

    return all_states, all_policies, all_outcomes, all_traces


# ═══════════════════════════════════════════
# § Evaluation Integration (Glicko-2 + PromotionGate)
# ═══════════════════════════════════════════

try:
    from quartz.evaluation import (
        TrainingEvaluator, EvalConfig, PromotionVerdict,
        RatingLadder, PromotionGate, ChampionTracker,
        RandomEngine as EvalRandomEngine, MatchRunner, tally_match,
        GameAdapter, Engine as EvalEngine, GameRecord,
    )
    HAS_EVAL_SYSTEM = True
except ImportError:
    HAS_EVAL_SYSTEM = False

class TreeMCTSEngine:
    """Wraps TreeMCTS to conform to calibration_eval.Engine protocol."""
    def __init__(self, engine_name, cfg, model, device):
        self._name = engine_name
        self._mcts = TreeMCTS(cfg, model, device)
        self._cfg = cfg
        self._eval_iters = self._cfg['iters'] // 4
    def select_move(self, state):
        board_enc = state._encode()
        # GomokuGameAdapter.current_player() returns 0/1 (for calibration_eval protocol)
        # TreeMCTS.search() expects 1/-1 (for board encoding)
        raw_player = state.current_player()  # 0 = black, 1 = white
        player = 1 if raw_player == 0 else -1  # convert to 1/-1
        legal_mask = np.zeros(self._cfg['actions'], dtype=np.float32)
        for action in state.legal_moves():
            if 0 <= action < self._cfg['actions']:
                legal_mask[action] = 1.0
        policy = self._mcts.search(board_enc, player, legal_mask, self._eval_iters)
        legal = state.legal_moves()
        if legal:
            chosen = max(legal, key=lambda a: policy[a] if a < len(policy) else 0)
        else:
            chosen = 0
        return chosen, {"time_used_ms": 0, "simulations": self._eval_iters}
    def reset(self): pass
    def name(self): return self._name


class RustNNEvaluatorEngine:
    """Evaluator engine using full Rust MCTS + NN for promotion evaluation.
    
    Uses the same Rust search stack as training self-play (TT+VL+PW+QUARTZ),
    ensuring evaluation semantics match training semantics.
    """
    def __init__(self, engine_name, cfg, model, device,
                 rust_binary="./target/release/mcts_demo"):
        self._name = engine_name
        self._cfg = cfg
        self._model = model
        self._device = device
        self._rust_binary = rust_binary
        self._client = None
        self._simulations = self._cfg.get('iters', 200)

    def _ensure_client(self):
        if self._client is None:
            self._client = NNSearchClient(self._model, self._cfg, self._device, self._rust_binary)
            self._client.start()

    def select_move(self, state):
        return self.select_moves_batch([state])[0]

    def select_moves_batch(self, states):
        self._ensure_client()
        penalty_mode = self._cfg.get('penalty_mode', 'GatedRefresh')
        game_name = self._cfg.get('_name')
        if is_chess_game(game_name):
            jobs = []
            players = []
            for state in states:
                raw_player = state.current_player()
                player = 1 if raw_player == 1 else -1
                players.append(player)
                job = {
                    "fen": getattr(state, "_fen", ""),
                    "player": int(player),
                }
                job.update(build_rust_state_meta(game_name, state, self._cfg))
                jobs.append(job)
            results = self._client.search_moves_multi(jobs, penalty_mode=penalty_mode)
            parsed = []
            for state, player, result in zip(states, players, results):
                if not result or 'error' in result:
                    raise RuntimeError(
                        f"rust_nn chess eval failed: {result.get('error', 'empty response') if isinstance(result, dict) else 'empty response'}")
                pol_entries = result.get('policy', [])
                if not pol_entries:
                    terminal_value = float(result.get('value', 0.0))
                    outcome_for_black = -terminal_value * player
                    parsed.append((0, {
                        "time_used_ms": 0,
                        "simulations": self._simulations,
                        "p_flip": result.get('p_flip', 0),
                        "engine": "rust_nn",
                        "terminal": True,
                        "outcome_for_black": float(outcome_for_black),
                    }))
                    continue
                best_move = int(result.get('best_move', 0))
                parsed.append((best_move, {
                    "time_used_ms": 0,
                    "simulations": self._simulations,
                    "p_flip": result.get('p_flip', 0),
                    "engine": "rust_nn",
                    "result_fen": result.get('result_fen', ''),
                    "result_history_hashes": result.get('result_history_hashes', []),
                }))
            return parsed

    def play_match_tally_against(self, opponent, game_factory, opening_book, num_games,
                                 color_swap=True, logger=None, max_moves=500, seed=None):
        if not isinstance(opponent, RustNNEvaluatorEngine):
            raise TypeError("shared Rust evaluation requires RustNNEvaluatorEngine opponent")
        shared_client = NNSearchClient(
            {0: self._model, 1: opponent._model},
            self._cfg,
            self._device,
            self._rust_binary,
        )
        shared_client.start()
        rng = random.Random(seed)
        sessions = []
        ob_n = len(opening_book) if opening_book else 0
        game_name = self._cfg.get('_name')
        progress_every = max(1, min(25, num_games // 10 if num_games > 0 else 1))
        completed_games = 0
        eval_loop_idx = 0
        stall_timeout_s = float(os.environ.get("QUARTZ_EVAL_STALL_TIMEOUT_S", "0") or 0.0)
        last_progress_sig = None
        last_progress_ts = time.time()

        def report_progress(force=False):
            if force or (completed_games > 0 and completed_games % progress_every == 0):
                print(f"  EvalProgress: {completed_games}/{num_games}", flush=True)

        def build_job(sess):
            game = sess["game"]
            mover_tag = sess["black_tag"] if game.current_player() == 0 else sess["white_tag"]
            player = 1 if game.current_player() == 0 else -1
            if is_chess_game(game_name):
                job = {
                    "fen": getattr(game, "_fen", ""),
                    "player": int(player),
                    "model_tag": int(mover_tag),
                }
            else:
                job = {
                    "board": list(getattr(game, "_board", [])),
                    "player": int(player),
                    "model_tag": int(mover_tag),
                }
            job.update(build_rust_state_meta(game_name, game, self._cfg))
            return job

        def apply_result(sess, result, fallback_ms):
            game = sess["game"]
            if not result or 'error' in result:
                sess["error"] = result.get("error", "empty response") if isinstance(result, dict) else "empty response"
                sess["done"] = True
                return
            # Terminal state: empty policy means the game is over (mate/stalemate).
            # Mark the session as done without trying to apply a move.
            pol_entries = result.get("policy", [])
            if not pol_entries or game.is_terminal():
                sess["done"] = True
                return
            move_time_ms = float(result.get("time_used_ms", 0.0) or 0.0)
            sess["total_time_ms"] += move_time_ms if move_time_ms > 0.0 else fallback_ms
            action = int(result.get("best_move", 0))
            meta = {
                "time_used_ms": move_time_ms if move_time_ms > 0.0 else fallback_ms,
                "simulations": int(result.get("iterations", self._simulations) or self._simulations),
                "p_flip": result.get("p_flip", 0),
                "engine": "rust_nn_shared_eval",
            }
            if hasattr(game, "apply_engine_meta") and result.get("result_fen"):
                meta["result_fen"] = result.get("result_fen", "")
                meta["result_history_hashes"] = result.get("result_history_hashes", [])
            applied = False
            if hasattr(game, "apply_engine_meta"):
                applied = bool(game.apply_engine_meta(action, meta))
            if not applied:
                try:
                    game.apply_move(action)
                except Exception as exc:
                    sess["error"] = str(exc)
                    sess["done"] = True
                    return
            sess["ply"] += 1
            if game.is_terminal() or sess["ply"] >= max_moves:
                sess["done"] = True

        def append_session(eng_black, eng_white, black_tag, white_tag, game_id, opening_idx=None):
            game = game_factory()
            opening_applied = []
            game_seed = rng.randint(0, 2**31)
            if opening_idx is not None and opening_idx < len(opening_book):
                for action in opening_book[opening_idx]:
                    if game.is_terminal() or action not in game.legal_moves():
                        break
                    game.apply_move(action)
                    opening_applied.append(action)
            sessions.append({
                "game_id": game_id,
                "game": game,
                "eng_black": eng_black,
                "eng_white": eng_white,
                "black_tag": int(black_tag),
                "white_tag": int(white_tag),
                "opening": opening_applied,
                "seed": game_seed,
                "ply": len(opening_applied),
                "total_time_ms": 0.0,
                "done": bool(game.is_terminal()),
                "error": None,
            })

        pairs = num_games // 2 if color_swap else 0
        for i in range(pairs):
            opening_idx = i % ob_n if ob_n else None
            append_session(self, opponent, 0, 1, f"g{2*i:04d}", opening_idx)
            append_session(opponent, self, 1, 0, f"g{2*i+1:04d}", opening_idx)
        for idx in range(2 * pairs, num_games):
            opening_idx = idx % ob_n if ob_n else None
            append_session(self, opponent, 0, 1, f"g{idx:04d}", opening_idx)

        records = []
        session_id = None
        use_rust_eval_runner = (
            self._cfg.get("_eval_runner_mode") == "rust_eval_state_machine"
            and supports_rust_eval_state_machine(game_name)
        )
        try:
            if use_rust_eval_runner:
                try:
                    runner_sessions = []
                    for sess in sessions:
                        payload = build_job(sess)
                        payload.update({
                            "game_id": sess["game_id"],
                            "black_tag": int(sess["black_tag"]),
                            "white_tag": int(sess["white_tag"]),
                            "opening": list(sess["opening"]),
                            "seed": int(sess["seed"]),
                            "ply": int(sess["ply"]),
                            "done": bool(sess["done"]),
                            "total_time_ms": float(sess["total_time_ms"]),
                        })
                        runner_sessions.append(payload)
                    _stall_trace(
                        "eval_runner_start",
                        game=game_name,
                        num_games=int(num_games),
                        runner_mode="rust_eval_state_machine",
                    )
                    payload = shared_client.eval_match_run(
                        runner_sessions,
                        max_moves=max_moves,
                        penalty_mode=self._cfg.get("penalty_mode", "GatedRefresh"),
                    )
                    raw_records = payload.get("records", []) if isinstance(payload, dict) else []
                    if isinstance(payload, dict) and payload.get("error"):
                        raise RuntimeError(str(payload.get("error")))
                    if not isinstance(raw_records, list) or len(raw_records) != len(sessions):
                        raise RuntimeError(
                            f"rust eval runner record length mismatch: expected {len(sessions)} got "
                            f"{len(raw_records) if isinstance(raw_records, list) else 'non-list'}"
                        )
                    for rec_data in raw_records:
                        black_tag = int(rec_data.get("black_tag", 0) or 0)
                        white_tag = int(rec_data.get("white_tag", 1) or 1)
                        rec = GameRecord(
                            game_id=str(rec_data.get("game_id", "")),
                            engine_black=self.name() if black_tag == 0 else opponent.name(),
                            engine_white=self.name() if white_tag == 0 else opponent.name(),
                            outcome=str(rec_data.get("outcome", "draw")),
                            score_black=rec_data.get("score_black", 0.5),
                            move_count=int(rec_data.get("move_count", 0) or 0),
                            total_time_ms=float(rec_data.get("total_time_ms", 0.0) or 0.0),
                            moves=[],
                            opening=list(rec_data.get("opening", []) or []),
                            seed=rec_data.get("seed"),
                            error=rec_data.get("error"),
                            is_void=bool(rec_data.get("is_void", False)),
                        )
                        records.append(rec)
                        if logger is not None:
                            logger.log(rec)
                        completed_games += 1
                        _stall_trace(
                            "eval_game_done",
                            game=game_name,
                            completed_games=int(completed_games),
                            total_games=int(num_games),
                            move_count=int(rec.move_count),
                            has_error=bool(rec.error),
                            runner_mode="rust_eval_state_machine",
                        )
                    report_progress(force=True)
                    return tally_match(records, self.name())
                except Exception as exc:
                    records.clear()
                    completed_games = 0
                    _stall_trace(
                        "eval_runner_fallback",
                        game=game_name,
                        num_games=int(num_games),
                        error=str(exc),
                    )
            session_supported = True
            if session_supported:
                payload = shared_client.open_search_session(
                    [build_job(sess) for sess in sessions],
                    penalty_mode=self._cfg.get('penalty_mode', 'GatedRefresh'),
                )
                session_id = payload.get("session_id") if isinstance(payload, dict) else None
                results = payload.get("results", []) if isinstance(payload, dict) else []
                _stall_trace(
                    "eval_session_open",
                    game=game_name,
                    num_games=int(num_games),
                    session_id=int(session_id) if session_id is not None else None,
                    result_count=int(len(results)) if isinstance(results, list) else None,
                )
            else:
                results = None

            while True:
                active = [sess for sess in sessions if not sess["done"] and not sess["game"].is_terminal() and sess["ply"] < max_moves]
                if not active:
                    break
                eval_loop_idx += 1
                progress_sig = (
                    int(len(active)),
                    int(sum(sess["ply"] for sess in sessions)),
                    int(sum(1 for sess in sessions if sess["done"])),
                    int(sum(1 for sess in sessions if sess["error"])),
                )
                if progress_sig != last_progress_sig:
                    last_progress_sig = progress_sig
                    last_progress_ts = time.time()
                elif stall_timeout_s > 0.0 and (time.time() - last_progress_ts) > stall_timeout_s:
                    _stall_trace(
                        "eval_stall",
                        game=game_name,
                        loop=int(eval_loop_idx),
                        active_games=int(len(active)),
                        total_ply=int(sum(sess["ply"] for sess in sessions)),
                        done_games=int(sum(1 for sess in sessions if sess["done"])),
                        error_games=int(sum(1 for sess in sessions if sess["error"])),
                        session_id=int(session_id) if session_id is not None else None,
                    )
                    raise RuntimeError(
                        f"evaluation stalled for {time.time() - last_progress_ts:.1f}s "
                        f"(active={len(active)} ply={sum(sess['ply'] for sess in sessions)})"
                    )
                _stall_trace(
                    "eval_loop",
                    game=game_name,
                    loop=int(eval_loop_idx),
                    active_games=int(len(active)),
                    done_games=int(sum(1 for sess in sessions if sess["done"])),
                    total_ply=int(sum(sess["ply"] for sess in sessions)),
                    session_mode=bool(session_id is not None),
                )

                if session_id is None:
                    jobs = [build_job(sess) for sess in active]
                    t0 = time.time()
                    results = shared_client.search_moves_multi(jobs, penalty_mode=self._cfg.get('penalty_mode', 'GatedRefresh'))
                    batch_elapsed_ms = max(0.0, (time.time() - t0) * 1000.0)
                    share_ms = batch_elapsed_ms / max(1, len(active))
                    if not isinstance(results, list) or len(results) != len(active):
                        for sess in active:
                            sess["error"] = (
                                f"shared eval result length mismatch: expected {len(active)} got "
                                f"{len(results) if isinstance(results, list) else 'non-list'}"
                            )
                            sess["done"] = True
                        break
                    for sess, result in zip(active, results):
                        apply_result(sess, result, share_ms)
                else:
                    t0 = time.time()
                    if not isinstance(results, list) or len(results) != len(sessions):
                        for sess in active:
                            sess["error"] = (
                                f"shared eval session result length mismatch: expected {len(sessions)} got "
                                f"{len(results) if isinstance(results, list) else 'non-list'}"
                            )
                            sess["done"] = True
                        break
                    share_ms = max(0.0, (time.time() - t0) * 1000.0) / max(1, len(active))
                    updates = []
                    for idx, sess in enumerate(sessions):
                        if sess["done"] or sess["game"].is_terminal() or sess["ply"] >= max_moves:
                            sess["done"] = True
                            updates.append({"deactivate": True})
                            continue
                        apply_result(sess, results[idx], share_ms)
                        if sess["done"]:
                            updates.append({"deactivate": True})
                        else:
                            updates.append({"action": int(results[idx].get("best_move", 0))})
                    payload = shared_client.step_search_session(session_id, updates)
                    results = payload.get("results", []) if isinstance(payload, dict) else []
                    _stall_trace(
                        "eval_session_step",
                        game=game_name,
                        loop=int(eval_loop_idx),
                        session_id=int(session_id),
                        updates=int(len(updates)),
                        result_count=int(len(results)) if isinstance(results, list) else None,
                    )

            for sess in sessions:
                game = sess["game"]
                if game.is_terminal():
                    if hasattr(game, "is_void_result") and game.is_void_result():
                        rec = GameRecord(
                            game_id=sess["game_id"],
                            engine_black=sess["eng_black"].name(),
                            engine_white=sess["eng_white"].name(),
                            outcome="void",
                            score_black=None,
                            move_count=sess["ply"],
                            total_time_ms=sess["total_time_ms"],
                            moves=[],
                            opening=sess["opening"],
                            seed=sess["seed"],
                            error=sess["error"],
                            is_void=True,
                        )
                    else:
                        outcome_for_black = float(game.outcome_for_black() or 0.0)
                        if outcome_for_black > 0:
                            outcome, score_black = "black_win", 1.0
                        elif outcome_for_black < 0:
                            outcome, score_black = "white_win", 0.0
                        else:
                            outcome, score_black = "draw", 0.5
                        rec = GameRecord(
                            game_id=sess["game_id"],
                            engine_black=sess["eng_black"].name(),
                            engine_white=sess["eng_white"].name(),
                            outcome=outcome,
                            score_black=score_black,
                            move_count=sess["ply"],
                            total_time_ms=sess["total_time_ms"],
                            moves=[],
                            opening=sess["opening"],
                            seed=sess["seed"],
                            error=sess["error"],
                            is_void=bool(sess["error"]),
                        )
                else:
                    rec = GameRecord(
                        game_id=sess["game_id"],
                        engine_black=sess["eng_black"].name(),
                        engine_white=sess["eng_white"].name(),
                        outcome="draw",
                        score_black=0.5,
                        move_count=sess["ply"],
                        total_time_ms=sess["total_time_ms"],
                        moves=[],
                        opening=sess["opening"],
                        seed=sess["seed"],
                        error=sess["error"],
                        is_void=bool(sess["error"]),
                    )
                records.append(rec)
                if logger is not None:
                    logger.log(rec)
                completed_games += 1
                _stall_trace(
                    "eval_game_done",
                    game=game_name,
                    completed_games=int(completed_games),
                    total_games=int(num_games),
                    move_count=int(sess["ply"]),
                    has_error=bool(sess["error"]),
                )
                report_progress()
        finally:
            if session_id is not None:
                try:
                    shared_client.close_search_session(session_id)
                except Exception:
                    pass
            shared_client.stop()
        report_progress(force=True)
        return tally_match(records, self.name())

        jobs = []
        legals = []
        for state in states:
            raw_player = state.current_player()
            player = 1 if raw_player == 0 else -1
            legal = state.legal_moves()
            legals.append(legal)
            jobs.append({
                "board": list(state._board),
                "player": int(player),
                **build_rust_state_meta(game_name, state, self._cfg),
            })
        results = self._client.search_moves_multi(jobs, penalty_mode=penalty_mode)
        parsed = []
        for legal, result in zip(legals, results):
            if not result or 'error' in result:
                parsed.append(((legal[0] if legal else 0), {"time_used_ms": 0, "simulations": 0}))
                continue
            pol_entries = result.get('policy', [])
            if not legal:
                parsed.append((0, {
                    "time_used_ms": 0,
                    "simulations": self._simulations,
                    "p_flip": result.get('p_flip', 0),
                    "engine": "rust_nn",
                }))
                continue
            legal_set = set(legal)
            policy = {}
            for action, val in iter_sparse_policy_entries(pol_entries):
                if action in legal_set and action < self._cfg['actions']:
                    policy[action] = val
            chosen = legal[0]
            best_val = policy.get(chosen, 0.0)
            for action in legal[1:]:
                value = policy.get(action, 0.0)
                if value > best_val:
                    chosen = action
                    best_val = value
            parsed.append((chosen, {
                "time_used_ms": 0,
                "simulations": self._simulations,
                "p_flip": result.get('p_flip', 0),
                "engine": "rust_nn",
            }))
        return parsed

    def reset(self):
        if self._client:
            self._client.stop()
            self._client = None

    def name(self): return self._name

    def __del__(self):
        self.reset()

class GomokuGameAdapter:
    """Flat-board adapter for gomoku-style games and variants."""

    def __init__(self, board_size=7, win_len=4, encoder=None, variant="gomoku7"):
        self._bs = board_size
        self._wl = win_len
        self._variant = variant
        self._board = [0] * (board_size * board_size)
        self._player = 1
        self._terminal = False
        self._outcome = None
        self._encoder = encoder
        self._ch = encoder.n_channels if encoder else 3

    def clone(self):
        g = GomokuGameAdapter(self._bs, self._wl, self._encoder, self._variant)
        g._board = self._board[:]
        g._player = self._player
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def _line_count(self, action, dr, dc):
        r0, c0 = action // self._bs, action % self._bs
        cnt = 1
        for sign in (1, -1):
            nr, nc = r0 + sign * dr, c0 + sign * dc
            while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == self._player:
                cnt += 1
                nr += sign * dr
                nc += sign * dc
        return cnt

    def _line_ends(self, action, dr, dc):
        r0, c0 = action // self._bs, action % self._bs
        stone = self._player
        forward = backward = 0

        nr, nc = r0 + dr, c0 + dc
        while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == stone:
            forward += 1
            nr += dr
            nc += dc
        forward_blocked = not (0 <= nr < self._bs and 0 <= nc < self._bs) or self._board[nr * self._bs + nc] == -stone

        nr, nc = r0 - dr, c0 - dc
        while 0 <= nr < self._bs and 0 <= nc < self._bs and self._board[nr * self._bs + nc] == stone:
            backward += 1
            nr -= dr
            nc -= dc
        backward_blocked = not (0 <= nr < self._bs and 0 <= nc < self._bs) or self._board[nr * self._bs + nc] == -stone

        return 1 + forward + backward, forward_blocked, backward_blocked

    def _is_winning_move(self, action):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cnt, forward_blocked, backward_blocked = self._line_ends(action, dr, dc)
            if self._variant == "gomoku15_std":
                if cnt == self._wl:
                    return True
                continue
            if self._variant in {"gomoku15_omok", "gomoku15_renju"}:
                if self._player == 1:
                    if cnt == self._wl:
                        return True
                elif cnt >= self._wl:
                    return True
                continue
            if self._variant == "gomoku15_caro":
                if cnt == self._wl and not (forward_blocked and backward_blocked):
                    return True
                continue
            if cnt >= self._wl:
                return True
        return False

    def apply_move(self, action):
        self._board[action] = self._player
        if self._is_winning_move(action):
            self._terminal = True
            self._outcome = 1.0 if self._player == 1 else -1.0
            self._player = -self._player
            return
        move_limit = 200 if self._variant == "gomoku15_renju" else self._bs * self._bs
        played = sum(1 for value in self._board if value != 0)
        if played >= move_limit or not any(b == 0 for b in self._board):
            self._terminal = True
            self._outcome = 0.0
        self._player = -self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        return [i for i, value in enumerate(self._board) if value == 0]

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        enc = np.zeros((self._ch, self._bs, self._bs), dtype=np.float32)
        for i in range(self._bs * self._bs):
            r, c = i // self._bs, i % self._bs
            if self._board[i] == self._player:
                enc[0, r, c] = 1.0
            elif self._board[i] != 0:
                enc[1, r, c] = 1.0
        if self._ch >= 3 and self._player == 1:
            enc[2] = 1.0
        return enc


class TicTacToeGameAdapter:
    def __init__(self, encoder=None):
        self._bs = 3
        self._board = [0] * 9
        self._player = 1
        self._terminal = False
        self._outcome = None
        self._encoder = encoder

    def clone(self):
        g = TicTacToeGameAdapter(self._encoder)
        g._board = self._board[:]
        g._player = self._player
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def apply_move(self, action):
        self._board[action] = self._player
        lines = (
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6),
        )
        for a, b, c in lines:
            if self._board[a] != 0 and self._board[a] == self._board[b] == self._board[c]:
                self._terminal = True
                self._outcome = 1.0 if self._player == 1 else -1.0
                self._player = -self._player
                return
        if not any(v == 0 for v in self._board):
            self._terminal = True
            self._outcome = 0.0
        self._player = -self._player

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        return [i for i, value in enumerate(self._board) if value == 0]

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        enc = np.zeros((3, 3, 3), dtype=np.float32)
        for i, value in enumerate(self._board):
            r, c = divmod(i, 3)
            if value == self._player:
                enc[0, r, c] = 1.0
            elif value != 0:
                enc[1, r, c] = 1.0
        if self._player == 1:
            enc[2] = 1.0
        return enc


class GoGameAdapter:
    """Local Go state used by training/eval for configurable Go rulesets."""

    def __init__(self, board_size=9, komi=7.5, encoder=None, scoring="area",
                 allow_suicide=False, ruleset="chinese"):
        self._bs = board_size
        self._board = [0] * (board_size * board_size)
        self._player = 1
        self._passes = 0
        self._ko_point = None
        self._terminal = False
        self._outcome = None
        self._komi = komi
        self._encoder = encoder
        self._scoring = scoring
        self._allow_suicide = allow_suicide
        self._ruleset = ruleset
        self._black_caps = 0
        self._white_caps = 0
        self._cycle_terminal = False
        self._history_hashes = {self._position_hash()}
        self._void_result = False

    def clone(self):
        g = GoGameAdapter(
            self._bs, self._komi, self._encoder,
            scoring=self._scoring,
            allow_suicide=self._allow_suicide,
            ruleset=self._ruleset)
        g._board = self._board[:]
        g._player = self._player
        g._passes = self._passes
        g._ko_point = self._ko_point
        g._terminal = self._terminal
        g._outcome = self._outcome
        g._black_caps = self._black_caps
        g._white_caps = self._white_caps
        g._cycle_terminal = self._cycle_terminal
        g._history_hashes = set(self._history_hashes)
        g._void_result = self._void_result
        return g

    def _position_hash(self, board=None, player=None):
        state_board = tuple(self._board if board is None else board)
        state_player = self._player if player is None else player
        return (state_board, state_player)

    def _neighbors(self, pos):
        r, c = divmod(pos, self._bs)
        if r > 0:
            yield pos - self._bs
        if r + 1 < self._bs:
            yield pos + self._bs
        if c > 0:
            yield pos - 1
        if c + 1 < self._bs:
            yield pos + 1

    def _group_and_liberties(self, board, pos):
        color = board[pos]
        if color == 0:
            return [], set()
        group = []
        liberties = set()
        stack = [pos]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            for nb in self._neighbors(cur):
                if board[nb] == 0:
                    liberties.add(nb)
                elif board[nb] == color and nb not in visited:
                    stack.append(nb)
        return group, liberties

    def _is_legal(self, pos):
        if self._cycle_terminal:
            return False
        if pos < 0 or pos >= len(self._board):
            return False
        if self._board[pos] != 0:
            return False
        if self._ko_point == pos:
            return False
        board = self._board[:]
        board[pos] = self._player
        opp = -self._player
        captured = 0
        for nb in self._neighbors(pos):
            if board[nb] == opp:
                group, liberties = self._group_and_liberties(board, nb)
                if not liberties:
                    captured += len(group)
                    for stone in group:
                        board[stone] = 0
        _, liberties = self._group_and_liberties(board, pos)
        if not (liberties or captured > 0 or self._allow_suicide):
            return False
        if self._ruleset == "chinese":
            next_player = -self._player
            if self._position_hash(board=board, player=next_player) in self._history_hashes:
                return False
        return True

    def _score(self):
        if self._scoring == "area":
            black = 0.0
            white = self._komi
            visited = set()
            for pos, value in enumerate(self._board):
                if value == 1:
                    black += 1.0
                elif value == -1:
                    white += 1.0
                elif pos not in visited:
                    region = []
                    owners = set()
                    stack = [pos]
                    while stack:
                        cur = stack.pop()
                        if cur in visited or self._board[cur] != 0:
                            continue
                        visited.add(cur)
                        region.append(cur)
                        for nb in self._neighbors(cur):
                            if self._board[nb] == 0:
                                stack.append(nb)
                            else:
                                owners.add(self._board[nb])
                    if owners == {1}:
                        black += float(len(region))
                    elif owners == {-1}:
                        white += float(len(region))
            return black, white

        board = self._board[:]
        black = float(self._black_caps)
        white = float(self._white_caps) + self._komi

        def classify_empty_regions(state_board):
            region_ids = {}
            owners = []
            for pos, value in enumerate(state_board):
                if value != 0 or pos in region_ids:
                    continue
                rid = len(owners)
                stack = [pos]
                region_ids[pos] = rid
                border = set()
                while stack:
                    cur = stack.pop()
                    for nb in self._neighbors(cur):
                        if state_board[nb] == 0 and nb not in region_ids:
                            region_ids[nb] = rid
                            stack.append(nb)
                        elif state_board[nb] != 0:
                            border.add(state_board[nb])
                if border == {1}:
                    owners.append(1)
                elif border == {-1}:
                    owners.append(-1)
                else:
                    owners.append(0)
            return region_ids, owners

        if self._ruleset in {"japanese", "korean"}:
            while True:
                region_ids, owners = classify_empty_regions(board)
                visited = set()
                removed_any = False
                for pos, color in enumerate(board):
                    if color == 0 or pos in visited:
                        continue
                    stack = [pos]
                    group = []
                    adj_regions = set()
                    touches_opponent = False
                    touches_edge = False
                    while stack:
                        cur = stack.pop()
                        if cur in visited:
                            continue
                        visited.add(cur)
                        group.append(cur)
                        row, col = divmod(cur, self._bs)
                        if row == 0 or row + 1 == self._bs or col == 0 or col + 1 == self._bs:
                            touches_edge = True
                        for nb in self._neighbors(cur):
                            if board[nb] == color:
                                stack.append(nb)
                            elif board[nb] == 0 and nb in region_ids:
                                adj_regions.add(region_ids[nb])
                            elif board[nb] == -color:
                                touches_opponent = True
                    eye_count = 0
                    touches_neutral = False
                    for rid in adj_regions:
                        owner = owners[rid]
                        if owner == color:
                            eye_count += 1
                        elif owner == 0:
                            touches_neutral = True
                    if eye_count < 2 and not touches_neutral and touches_opponent and not touches_edge:
                        removed_any = True
                        for stone in group:
                            board[stone] = 0
                        if color == 1:
                            white += float(len(group))
                        else:
                            black += float(len(group))
                if not removed_any:
                    break

        visited = set()
        for pos, value in enumerate(board):
            if value != 0 or pos in visited:
                continue
            region = []
            owners = set()
            stack = [pos]
            while stack:
                cur = stack.pop()
                if cur in visited or board[cur] != 0:
                    continue
                visited.add(cur)
                region.append(cur)
                for nb in self._neighbors(cur):
                    if board[nb] == 0:
                        stack.append(nb)
                    else:
                        owners.add(board[nb])
            if owners == {1}:
                black += float(len(region))
            elif owners == {-1}:
                white += float(len(region))
        return black, white

    def apply_move(self, action):
        pass_action = self._bs * self._bs
        if action == pass_action:
            self._passes += 1
            self._ko_point = None
            self._player = -self._player
            if self._ruleset in {"japanese", "korean"} and self._passes < 2 and self._position_hash() in self._history_hashes:
                self._cycle_terminal = True
            self._history_hashes.add(self._position_hash())
        else:
            self._passes = 0
            self._board[action] = self._player
            opp = -self._player
            captured_points = []
            for nb in self._neighbors(action):
                if self._board[nb] == opp:
                    group, liberties = self._group_and_liberties(self._board, nb)
                    if not liberties:
                        captured_points.extend(group)
            for stone in captured_points:
                self._board[stone] = 0
            if captured_points:
                if self._player == 1:
                    self._black_caps += len(captured_points)
                else:
                    self._white_caps += len(captured_points)
            group, liberties = self._group_and_liberties(self._board, action)
            self._ko_point = None
            if len(captured_points) == 1 and len(liberties) == 1:
                self._ko_point = captured_points[0]
            if not liberties:
                if self._allow_suicide:
                    for stone in group:
                        self._board[stone] = 0
                    if self._player == 1:
                        self._white_caps += len(group)
                    else:
                        self._black_caps += len(group)
                else:
                    raise ValueError(f"illegal suicide move at {action}")
            self._player = -self._player
            if self._ruleset in {"japanese", "korean"} and self._position_hash() in self._history_hashes:
                self._cycle_terminal = True
            self._history_hashes.add(self._position_hash())
        if self._passes >= 2 or self._cycle_terminal:
            self._terminal = True
            if self._cycle_terminal:
                self._void_result = self._ruleset == "japanese"
                self._outcome = None if self._void_result else 0.0
            else:
                self._void_result = False
                black, white = self._score()
                if black > white:
                    self._outcome = 1.0
                elif white > black:
                    self._outcome = -1.0
                else:
                    self._outcome = 0.0

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def is_void_result(self):
        return self._terminal and self._void_result

    def current_player(self):
        return 0 if self._player == 1 else 1

    def legal_moves(self):
        if self._terminal:
            return []
        moves = [i for i, value in enumerate(self._board) if value == 0 and self._is_legal(i)]
        moves.append(self._bs * self._bs)
        return moves

    def _encode(self):
        if self._encoder is not None:
            return self._encoder.encode(np.array(self._board, dtype=np.int8), self._player)
        return encode_board({"board": self._bs, "ch": 17}, np.array(self._board, dtype=np.int8), self._player)


class ChessEvaluationAdapter:
    """Engine-driven chess state for evaluator matches.

    Rust search owns legality and state transitions. This adapter only stores the
    current FEN and terminal outcome so the generic evaluator can reuse the same
    match/glicko pipeline as other games.
    """

    supports_random_baseline = False

    def __init__(self, actions=CHESS_POLICY_ACTIONS, encoder=None, start_fen=STANDARD_CHESS_FEN):
        self._actions = actions
        self._encoder = encoder
        self._fen = start_fen
        self._chess_history_hashes = None
        self._terminal = False
        self._outcome = None

    def clone(self):
        g = ChessEvaluationAdapter(self._actions, self._encoder, self._fen)
        g._chess_history_hashes = (
            list(self._chess_history_hashes) if self._chess_history_hashes is not None else None
        )
        g._terminal = self._terminal
        g._outcome = self._outcome
        return g

    def _side_part(self):
        parts = self._fen.split()
        return parts[1] if len(parts) >= 2 else "w"

    def current_player(self):
        return 1 if self._side_part() == "w" else 0

    def legal_moves(self):
        if self._terminal:
            return []
        return list(range(self._actions))

    def apply_engine_meta(self, action, meta):
        if meta.get("terminal", False):
            self._terminal = True
            self._outcome = float(meta.get("outcome_for_black", 0.0))
            return True
        new_fen = meta.get("result_fen", "")
        if not new_fen or new_fen == self._fen:
            return False
        self._fen = new_fen
        history_hashes = meta.get("result_history_hashes")
        if history_hashes is not None:
            self._chess_history_hashes = [int(v) for v in history_hashes]
        return True

    def apply_move(self, action):
        raise RuntimeError("Chess evaluator requires engine-provided state transitions")

    def is_terminal(self):
        return self._terminal

    def outcome_for_black(self):
        return self._outcome

    def _encode(self):
        if self._encoder is not None:
            return encode_chess_fen(self._fen)
        return encode_chess_fen(self._fen)


def build_training_game_adapter(cfg):
    game_name = cfg.get('_name')
    encoder = cfg.get('_encoder')
    if game_name == "tictactoe":
        return TicTacToeGameAdapter(encoder=encoder)
    if is_chess_game(game_name):
        return ChessEvaluationAdapter(
            actions=cfg.get('actions', CHESS_POLICY_ACTIONS),
            encoder=encoder,
            start_fen=initial_chess_fen(cfg))
    if is_go_game(game_name):
        return GoGameAdapter(
            board_size=cfg['board'],
            komi=cfg.get('go_komi', 7.5),
            encoder=encoder,
            scoring=cfg.get('go_scoring', 'area'),
            allow_suicide=cfg.get('go_allow_suicide', False),
            ruleset=cfg.get('go_ruleset', 'chinese'))
    if game_name in GOMOKU15_VARIANTS or game_name == "gomoku7":
        return GomokuGameAdapter(board_size=cfg['board'], win_len=cfg['win'], encoder=encoder, variant=game_name)
    raise ValueError(f"No local game adapter for {game_name}")


def main():
    parser = argparse.ArgumentParser(description="QUARTZ AlphaZero Training")
    parser.add_argument("--game", choices=list(GAME_CONFIGS.keys()), default="gomoku15")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--arena", nargs=2, metavar=("MODEL_A","MODEL_B"),
                        help="Compare two models head-to-head (e.g. --arena best_a.pt best_b.pt)")
    parser.add_argument("--arena-games", type=int, default=50)
    parser.add_argument("--arena-3agent", nargs=2, metavar=("CURRENT","BEST"),
                        help="3-agent arena: current model, best model, + random anchor")
    parser.add_argument("--concurrent", action="store_true", default=True,
                        help="Run self-play in background thread while training (default: on)")
    parser.add_argument("--no-pipeline", dest="concurrent", action="store_false",
                        help="Disable pipelined self-play (sequential mode for debugging)")
    parser.add_argument("--backend", choices=["auto","jax","torch"], default="auto",
                        help="ML backend: auto (prefer torch), jax (explicit opt-in), torch")
    parser.add_argument("--no-autotune", dest="autotune", action="store_false",
                        help="Disable hardware-based runtime autotuning")
    parser.add_argument("--retune", action="store_true",
                        help="Ignore saved autotune profile and rerun warmup benchmark")
    parser.add_argument("--rust-nn", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--replay-buffer", type=int, default=None,
                        help="Replay buffer capacity override")
    parser.add_argument("--replay-recent-frac", type=float, default=None,
                        help="Fraction of each training batch drawn from recent replay")
    parser.add_argument("--replay-window", type=int, default=None,
                        help="Recent replay sampling window size")
    parser.add_argument("--go-ruleset", choices=["chinese", "japanese", "korean"], default=None,
                        help="Override Go ruleset preset")
    parser.add_argument("--go-scoring", choices=["area", "territory"], default=None,
                        help="Override Go scoring mode")
    parser.add_argument("--go-komi", type=float, default=None,
                        help="Override Go komi")
    parser.add_argument("--go-allow-suicide", action="store_true",
                        help="Allow suicide moves in Go")
    parser.add_argument("--chess960-index", type=int, default=None,
                        help="Use a fixed Chess960 Scharnagl index (0-959)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--inner-patience", type=int, default=8,
                        help="Loose within-iteration early stopping patience (0 disables)")
    parser.add_argument("--inner-min-fraction", type=float, default=0.7,
                        help="Minimum fraction of planned train steps to run before inner stopping can trigger")
    parser.add_argument("--inner-min-delta", type=float, default=5e-4,
                        help="Minimum inner-step loss improvement to reset plateau tracking")
    parser.add_argument("--inner-ema-alpha", type=float, default=0.2,
                        help="EMA smoothing for inner-step plateau tracking")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Run Glicko-2 evaluation every N iterations")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="Number of games per evaluation")
    parser.add_argument("--selfplay-parallel", type=int, default=None,
                        help="Override foreground self-play process parallelism")
    parser.add_argument("--bg-parallel", type=int, default=None,
                        help="Override background self-play process parallelism")
    parser.add_argument("--bg-batch-games", type=int, default=None,
                        help="Override background self-play games per refill cycle")
    parser.add_argument("--mcts-threads", type=int, default=None,
                        help="Override Rust MCTS threads per self-play search")
    parser.add_argument("--nn-batch-size", type=int, default=None,
                        help="Override Rust->Python NN batch size")
    parser.add_argument("--search-profile", choices=["quartz", "baseline", "baseline_strict"], default=None,
                        help="Rust MCTS profile: quartz, baseline shared substrate, or baseline_strict")
    parser.add_argument("--vl-mode", choices=["disabled", "fixed", "adaptive", "vvisit_only", "vvalue_only"],
                        default=None, help="Virtual loss mode override for ablation study")
    parser.add_argument("--games", type=int, default=None, help="Self-play games per iteration override")
    parser.add_argument("--resident-session", action="store_true",
                        help="Experimental: enable Rust resident search sessions for self-play")
    parser.add_argument("--runtime-autotune", action="store_true",
                        help="Enable experimental online runtime retuning during training")
    parser.add_argument("--no-eval-selfplay-isolation", dest="eval_selfplay_isolation", action="store_false",
                        help="Allow background self-play to continue during evaluation")
    parser.set_defaults(eval_selfplay_isolation=True)
    args = parser.parse_args()
    if not hasattr(args, "autotune"):
        args.autotune = True

    # Fix 9: Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = dict(GAME_CONFIGS[args.game])
    cfg['_name'] = args.game
    cfg['_resident_session'] = bool(args.resident_session)

    # Attach game-agnostic encoder
    if get_encoder is not None:
        try:
            cfg['_encoder'] = get_encoder(args.game)
            print(f"  Encoder: {type(cfg['_encoder']).__name__} ({cfg['_encoder'].n_channels}ch, {cfg['_encoder'].n_actions} actions)")
        except KeyError:
            cfg['_encoder'] = None
    else:
        cfg['_encoder'] = None

    # Load config overrides
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            overrides = json.load(f)
            cfg = apply_config_overrides(cfg, overrides)

    if args.replay_buffer is not None:
        cfg['buf'] = max(1, int(args.replay_buffer))
    if args.replay_recent_frac is not None:
        cfg['recent_frac'] = float(max(0.0, min(1.0, args.replay_recent_frac)))
    if args.replay_window is not None:
        cfg['recent_window'] = max(0, int(args.replay_window))
    if is_go_game(args.game):
        if args.go_ruleset is not None:
            cfg['go_ruleset'] = args.go_ruleset
        if args.go_scoring is not None:
            cfg['go_scoring'] = args.go_scoring
        if args.go_komi is not None:
            cfg['go_komi'] = float(args.go_komi)
        if args.go_allow_suicide:
            cfg['go_allow_suicide'] = True
    if cfg.get("chess960", False) and args.chess960_index is not None:
        cfg['chess960_index'] = max(0, min(959, int(args.chess960_index)))
    if args.search_profile is not None:
        cfg["search_profile"] = args.search_profile
    if args.vl_mode is not None:
        cfg["vl_mode"] = args.vl_mode
    if args.games is not None:
        cfg["games"] = max(1, int(args.games))

    base_dir = args.output or default_output_dir(args.game)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    paths = resolve_runtime_paths(base_dir, explicit_model=args.model, resume=args.resume)
    model_path = paths["load_model_path"]
    latest_model_path = paths["latest_model_path"]
    best_model_path = paths["best_model_path"]
    replay_path = paths["replay_path"]
    log_path = paths["log_path"]
    autotune_profile_path = paths["autotune_profile_path"]

    # Device
    if args.device == "auto":
        device = torch.device(auto_device_name())
    else:
        device = torch.device(args.device)

    hw = detect_hardware_spec(device)
    configure_torch_rocm_runtime(hw)
    base_cfg = dict(cfg)
    eval_runner_mode = "python_batched"
    if os.path.exists(args.rust_binary):
        eval_runner_mode = (
            "rust_eval_state_machine"
            if supports_rust_eval_state_machine(cfg.get("_name"))
            else "shared_client_session"
        )
    selfplay_runner_mode = (
        "rust_selfplay_state_machine"
        if os.path.exists(args.rust_binary) and supports_rust_selfplay_state_machine(cfg.get("_name"))
        else "python_batched"
    )
    cfg["_selfplay_topology_version"] = 6
    cfg["_shared_eval_session"] = True
    cfg["_broker_enabled"] = False
    cfg["_eval_runner_mode"] = eval_runner_mode
    cfg["_selfplay_runner_mode"] = selfplay_runner_mode
    cfg["_runtime_tuner_enabled"] = bool(args.runtime_autotune)
    cfg["_eval_selfplay_isolated"] = bool(args.eval_selfplay_isolation)
    if args.autotune:
        cfg = autotune_training_cfg(cfg, hw, concurrent=args.concurrent)
        cfg['_resident_session'] = bool(args.resident_session)
    cfg["_selfplay_topology_version"] = 6
    cfg["_shared_eval_session"] = True
    cfg["_broker_enabled"] = False
    cfg["_eval_runner_mode"] = eval_runner_mode
    cfg["_selfplay_runner_mode"] = selfplay_runner_mode
    cfg = clamp_runtime_cfg_to_hardware(cfg, hw)
    try:
        if getattr(device, "type", str(device)) == "cpu":
            torch.set_num_threads(max_supported_threads(hw))
        else:
            torch.set_num_threads(gpu_host_thread_cap(hw))
    except Exception:
        pass
    try:
        if getattr(device, "type", str(device)) == "cpu":
            torch.set_num_interop_threads(max(1, min(max_supported_threads(hw), getattr(hw, "physical_cpus", 1) or 1)))
        else:
            torch.set_num_interop_threads(gpu_interop_thread_cap(hw))
    except Exception:
        pass

    # Model + Backend
    backend = None
    model = AlphaZeroNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = None

    if not args.serve and not args.arena:
        # Try unified backend for training (JAX JIT if available)
        try:
            from quartz.backend import create_backend
            backend = create_backend(cfg, device=args.device, preference=args.backend)
            if os.path.exists(model_path):
                backend.load(model_path)
            n_params_str = f"{n_params:,}"
            print(f"  Using {backend.name.upper()} backend ({n_params_str} params)")
        except Exception as e:
            if str(args.backend or "auto").lower() == "jax":
                print(
                    "  JAX backend requested explicitly, but initialization failed.\n"
                    "  On AMD Radeon Linux, current official JAX support is inference-only,\n"
                    "  so QUARTZ training with --backend jax is experimental.\n"
                    "  Use --backend torch for reliable local training."
                )
                raise
            print(f"  Backend init failed ({e}), using direct PyTorch")
            backend = None

    if backend is None:
        if os.path.exists(model_path):
            try:
                load_torch_state_dict_checked(model, model_path, torch, map_location=device)
                print(f"Loaded model: {model_path}")
            except Exception as exc:
                explicit_model = bool(args.model)
                if explicit_model:
                    raise
                print(f"  [WARN] Ignoring incompatible checkpoint {model_path} ({exc})")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
    actor_source = get_actor_model(model, backend)
    cfg["_backend_name"] = backend.name if backend is not None else "torch"

    benchmark_info = None
    if args.autotune and not args.serve and not args.arena and not args.arena_3agent:
        profile = None if args.retune else load_autotune_profile(autotune_profile_path, hw, cfg)
        if profile is not None:
            cfg = apply_runtime_overrides(cfg, profile.get("overrides", {}))
            cfg = clamp_runtime_cfg_to_hardware(cfg, hw)
            benchmark_info = profile.get("benchmarks", {})
            print("  Auto-tune profile: loaded cached benchmark")
        else:
            print("  Auto-tune profile: running warmup benchmark...")
            overrides, benchmark_info = run_autotune_benchmark(
                cfg, backend, actor_source, optimizer, device, hw, args.rust_binary,
                concurrent=args.concurrent)
            if overrides:
                cfg = apply_runtime_overrides(cfg, overrides)
                cfg = clamp_runtime_cfg_to_hardware(cfg, hw)
                save_autotune_profile(autotune_profile_path, hw, cfg, overrides, benchmark_info)
                print(f"  Auto-tune profile: saved to {autotune_profile_path}")
            else:
                print("  Auto-tune profile: benchmark produced no overrides")

    # Probe optimal inference batch_size (runs once during --retune)
    if args.autotune and not args.serve and not args.arena:
        _eval_batch_cap = cfg.get("_eval_batch_cap", 192)
        if _eval_batch_cap < 32:
            _eval_batch_cap = 32
        _probe_model = actor_source if not isinstance(actor_source, dict) else None
        if _probe_model is not None and not os.environ.get("QUARTZ_DISABLE_BATCH_PROBE"):
            try:
                probed_bs = _probe_inference_batch_size(_probe_model, device, cfg, _eval_batch_cap)
                if probed_bs > cfg.get("batch_size", 8):
                    cfg["batch_size"] = probed_bs
                    print(f"  Batch size probe: optimal BS={probed_bs} (cap={_eval_batch_cap})")
            except Exception as e:
                print(f"  Batch size probe: skipped ({e})")

    manual_runtime_overrides = {}
    if args.selfplay_parallel is not None:
        manual_runtime_overrides["selfplay_parallel"] = max(1, int(args.selfplay_parallel))
    if args.bg_parallel is not None:
        manual_runtime_overrides["bg_parallel"] = max(1, int(args.bg_parallel))
    if args.bg_batch_games is not None:
        manual_runtime_overrides["bg_batch_games"] = max(1, int(args.bg_batch_games))
    if args.mcts_threads is not None:
        manual_runtime_overrides["n_threads"] = clamp_thread_count(args.mcts_threads, hw)
    if args.nn_batch_size is not None:
        manual_runtime_overrides["batch_size"] = max(1, int(args.nn_batch_size))
    if manual_runtime_overrides:
        cfg = apply_runtime_overrides(cfg, manual_runtime_overrides)
        cfg = clamp_runtime_cfg_to_hardware(cfg, hw)

    if args.serve:
        serve(model, cfg, device)
        return

    if args.arena_3agent:
        rating_path = os.path.join(base_dir, "glicko2_ratings.json")
        ratings, promoted = arena_3agent(
            args.arena_3agent[0], args.arena_3agent[1], cfg, device,
            games_per_pair=args.arena_games // 3, rust_binary=args.rust_binary,
            use_rust_nn=args.rust_nn, rating_path=rating_path)
        return

    if args.arena:
        if args.rust_nn:
            print("  Arena mode: Rust MCTS + NN (full search stack)")
            wa, wb, d, wr, ci, sprt = arena_rust_nn(
                args.arena[0], args.arena[1], cfg, device, args.arena_games, args.rust_binary)
        else:
            print("  Arena mode: Python TreeMCTS")
            wa, wb, d, wr, ci, sprt = arena_compare(
                args.arena[0], args.arena[1], cfg, device, args.arena_games)
        print(f"Arena: A={wa} B={wb} D={d} | WinRate_A={wr:.3f} 95%CI=[{ci[0]:.3f},{ci[1]:.3f}] SPRT={sprt}")
        if sprt == "H1_accept": print("  → SPRT: Model A is significantly stronger (p<0.05)")
        elif sprt == "H0_accept": print("  → SPRT: No significant difference (p<0.05)")
        else: print(f"  → SPRT: Inconclusive after {args.arena_games} games")
        return

    if args.arena_3agent:
        from rating import RatingStore, round_robin_arena, Rating, ANCHOR_MU
        print(f"  3-Agent Arena: random + current + best")
        current_path, best_path = args.arena_3agent
        model_cur = AlphaZeroNet(cfg).to(device)
        model_cur.load_state_dict(load_torch_state_dict(current_path, torch, map_location=device))
        model_cur.eval()
        model_best = AlphaZeroNet(cfg).to(device)
        model_best.load_state_dict(load_torch_state_dict(best_path, torch, map_location=device))
        model_best.eval()

        mcts_random = TreeMCTS(cfg, model=None, device=device)
        mcts_cur = TreeMCTS(cfg, model=model_cur, device=device)
        mcts_best = TreeMCTS(cfg, model=model_best, device=device)

        def make_play_fn(board_size, n_actions, win_len, iters):
            def play(agent_a, agent_b, n_games):
                wa, wb, d = 0, 0, 0
                for gi in range(n_games):
                    first, second = (agent_a, agent_b) if gi%2==0 else (agent_b, agent_a)
                    first_is_a = (gi % 2 == 0)
                    board = np.zeros(board_size**2, dtype=np.int8)
                    player = 1; winner = 0
                    for _ in range(board_size**2):
                        enc = np.zeros((cfg['ch'], board_size, board_size), dtype=np.float32)
                        for i in range(board_size**2):
                            r,c = i//board_size, i%board_size
                            if board[i]==player: enc[0,r,c]=1.0
                            elif board[i]!=0: enc[1,r,c]=1.0
                        if cfg['ch']>=3 and player==1: enc[2]=1.0
                        legal_mask = np.array([1.0 if board[i]==0 else 0.0 for i in range(min(board_size**2,n_actions))])
                        if n_actions > board_size**2: legal_mask = np.concatenate([legal_mask, np.zeros(n_actions-board_size**2)])
                        mcts = first if player==1 else second
                        pol = mcts.search(enc, player, legal_mask, iters//4)
                        legal = [i for i in range(board_size**2) if board[i]==0]
                        if not legal: break
                        chosen = max(legal, key=lambda a: pol[a] if a<n_actions else 0)
                        board[chosen] = player
                        if win_len > 0:
                            r0,c0 = chosen//board_size, chosen%board_size
                            for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
                                cnt=1
                                for sign in [1,-1]:
                                    nr,nc=r0+sign*dr,c0+sign*dc
                                    while 0<=nr<board_size and 0<=nc<board_size and board[nr*board_size+nc]==player:
                                        cnt+=1;nr+=sign*dr;nc+=sign*dc
                                if cnt>=win_len: winner=player;break
                            if winner: break
                        player = -player
                    if winner==1:
                        if first_is_a: wa+=1
                        else: wb+=1
                    elif winner==-1:
                        if first_is_a: wb+=1
                        else: wa+=1
                    else: d+=1
                return wa, wb, d
            return play

        agents = {"random": mcts_random, "current": mcts_cur, "best": mcts_best}
        play_fn = make_play_fn(cfg['board'], cfg['actions'], cfg['win'], cfg['iters'])
        store_path = os.path.join(base_dir, "ratings.json")
        store = RatingStore(store_path)
        store.set("random", Rating(mu=ANCHOR_MU, phi=100.0, sigma=0.06))

        ratings = round_robin_arena(agents, play_fn, args.arena_games // 3, store)
        print(store.summary())
        print(f"\n  Ratings saved to {store_path}")
        return

    print(f"Game: {args.game} ({cfg['board']}×{cfg['board']})")
    print(f"Model: {n_params:,} params, {cfg['filters']}f×{cfg['blocks']}b")
    print(f"Backend: {backend.name if backend else 'PyTorch (direct)'}")
    print(f"Device: {device}")
    print(f"Output: {base_dir}")
    if cfg.get("search_profile", "quartz") != "quartz":
        print(f"Search profile: {cfg.get('search_profile')}")
    if cfg.get('_resident_session', False):
        print("  Self-play transport: experimental resident Rust session ENABLED")
    print_autotune_summary(base_cfg, cfg, hw)
    print(f"Replay sampling: {int(cfg.get('recent_frac', 0.0) * 100)}% recent,"
          f" window={cfg.get('recent_window', 0):,}")
    print(f"Replay buffer: capacity={cfg['buf']:,}, batch={cfg['batch']}")
    if is_go_game(args.game):
        print(
            "Go rules: "
            f"ruleset={cfg.get('go_ruleset', 'chinese')} "
            f"scoring={cfg.get('go_scoring', 'area')} "
            f"komi={cfg.get('go_komi', 7.5):.1f} "
            f"allow_suicide={bool(cfg.get('go_allow_suicide', False))}"
        )
    if cfg.get("chess960", False):
        start_desc = cfg.get("chess960_index")
        print(f"Chess960 start: {'randomized' if start_desc is None else f'index={start_desc}'}")

    # Replay buffer
    replay = ReplayBuffer(
        cfg['buf'],
        recent_fraction=cfg.get('recent_frac', 0.0),
        recent_window=cfg.get('recent_window', 0))
    if args.resume:
        n = replay.load(replay_path)
        if n: print(f"Loaded {n} positions from replay")
    outer_stopper = EarlyStopping(
        patience=args.patience,
        warmup=20 if args.concurrent else 10,
        ema_alpha=0.25 if args.concurrent else 0.30) if early_stopping_enabled(
            args.patience, concurrent=args.concurrent) else None
    inner_stop_cfg = {
        "patience": max(0, int(args.inner_patience)),
        "min_fraction": float(max(0.0, min(1.0, args.inner_min_fraction))),
        "min_delta": float(max(0.0, args.inner_min_delta)),
        "ema_alpha": float(max(0.01, min(1.0, args.inner_ema_alpha))),
    }
    log_f = open(log_path, "a")

    # ── Evaluation system (Glicko-2 + PromotionGate) ──
    training_evaluator = None
    eval_autotune_profile_path = os.path.join(base_dir, "eval_autotune.json")
    cached_eval_parallel_workers = load_eval_autotune_profile(
        eval_autotune_profile_path, hw, cfg, args.eval_games)
    eval_workers_autotuned = cached_eval_parallel_workers is not None
    if HAS_EVAL_SYSTEM:
        eval_parallel_workers = cached_eval_parallel_workers or recommend_eval_parallel_workers(
            hw, cfg, args.eval_games, rust_ok=os.path.exists(args.rust_binary))
        eval_parallel_workers = max(1, min(int(eval_parallel_workers), max_supported_threads(hw)))
        eval_cfg = EvalConfig(
            num_games=args.eval_games,
            promotion_threshold=0.55,
            confidence=0.95,
            sanity_check_interval=5,
            ladder_path=os.path.join(base_dir, "glicko2_ladder.json"),
            log_path=os.path.join(base_dir, "eval_matches.jsonl"),
            champion_path=os.path.join(base_dir, "champion.json"),
            seed=args.seed,
            parallel_workers=eval_parallel_workers,
        )
        game_factories = {
            game_name: (lambda game_name=game_name: build_training_game_adapter(
                dict(cfg, _name=game_name)))
            for game_name in GAME_CONFIGS
        }
        game_factory = game_factories.get(args.game)
        if game_factory:
            training_evaluator = TrainingEvaluator(config=eval_cfg)
            if not os.path.exists(best_model_path):
                if backend:
                    backend.save(best_model_path)
                elif model:
                    torch.save(model.state_dict(), best_model_path)
            else:
                ensure_best_checkpoint_compatible(best_model_path, backend, model, device)
            eval_worker_msg = str(eval_parallel_workers)
            if not eval_workers_autotuned:
                eval_worker_msg += " (first eval will benchmark)"
            print(
                f"  Eval system: Glicko-2 + PromotionGate "
                f"(every {args.eval_interval} iters, {args.eval_games} games, workers={eval_worker_msg})"
            )
        else:
            print(f"  Eval system: not available for {args.game}")

    # Check Rust binary
    rust_ok = os.path.exists(args.rust_binary)
    if not rust_ok:
        print(f"WARNING: Rust binary not found at {args.rust_binary}")
        print(f"  Training requires Rust. Run: cargo build --release")
    auto_resident_session = False
    cfg["_resident_session"] = bool(cfg.get("_resident_session", False) or auto_resident_session)

    print(f"\n{'='*60}")
    print(f"  Training: {args.iterations} iterations, {cfg['games']} games/iter")
    if args.concurrent: print(f"  Mode: CONCURRENT (background self-play)")
    print(f"  Runtime tuner: {'enabled' if args.runtime_autotune else 'disabled'}")
    if args.concurrent:
        print(f"  Eval/self-play isolation: {'enabled' if args.eval_selfplay_isolation else 'disabled'}")
    if outer_stopper is not None:
        print(f"  Outer early stopping: enabled (patience={args.patience}, warmup={outer_stopper.warmup})")
    if inner_stop_cfg["patience"] > 0:
        print(
            "  Inner early stopping: "
            f"enabled (patience={inner_stop_cfg['patience']}, min_fraction={inner_stop_cfg['min_fraction']:.2f})"
        )
    print(f"{'='*60}\n")

    # Start background self-play worker if concurrent
    bg_worker = None
    if args.concurrent:
        if not rust_ok:
            print("ERROR: --concurrent requires Rust binary. Run: cargo build --release")
            sys.exit(1)
        bg_worker = SelfPlayWorker(cfg, actor_source, device, replay, args.rust_binary)
        bg_worker.start()
        print("  [BG] Self-play worker started (Rust+NN)")
        # Wait for initial replay fill
        while len(replay) < initial_replay_fill_target(cfg, bg_worker._recent_chunks) and not (outer_stopper and outer_stopper.should_stop):
            time.sleep(0.5)
            bg_status = bg_worker.status()
            if not bg_status.get("alive", True):
                raise RuntimeError(
                    f"background self-play worker exited during replay fill: "
                    f"{bg_status.get('last_error') or 'thread exited'}"
                )
            if (
                bg_status.get("last_progress_age_s", 0.0) > bg_worker.REPLAY_STALL_TIMEOUT_S
                and bg_status.get("consecutive_errors", 0) > 0
            ):
                raise RuntimeError(
                    f"background self-play worker stalled during replay fill: "
                    f"{bg_status.get('last_error') or 'no progress'}"
                )
            status_suffix = ""
            if bg_status.get("consecutive_errors", 0) > 0:
                status_suffix = f" err={bg_status['consecutive_errors']}"
            fill_target = initial_replay_fill_target(cfg, bg_worker._recent_chunks)
            print(
                f"\r  [BG] Filling replay: {len(replay)}/{fill_target}..."
                f"{status_suffix}",
                end="",
                flush=True,
            )
        print()

    latest_eval = {
        "published_elo": None,
        "champion_elo": None,
        "elo_gap": None,
        "delta_elo": None,
        "score_rate": None,
        "eval_verdict": None,
    }
    online_tuner = OnlineAutotuneController(
        cfg, hw, enabled_iters=min(10, args.iterations), interval=2) if args.concurrent and args.runtime_autotune else None

    for iteration in range(args.iterations):
        clear_nn_eval_cache()  # invalidate after model update
        t0 = time.time()
        # LR schedule
        progress = iteration / max(args.iterations, 1)
        lr = 0.0002 + 0.5 * (0.02 - 0.0002) * (1 + math.cos(math.pi * progress))
        if optimizer:
            for pg in optimizer.param_groups: pg['lr'] = lr
        if backend: backend.set_lr(lr)
        avg_pflip = None
        should_early_stop = False
        entry = {
            "iter": iteration + 1,
            "loss": None,
            "p_loss": None,
            "v_loss": None,
            "loss_ema": round_or_none(outer_stopper.loss_ema if outer_stopper else None),
            "lr": round(lr, 6),
            "replay": len(replay),
            "new_pos": 0,
            "train_steps": 0,
            "planned_train_steps": 0,
            "time_s": None,
            "pos_per_s": None,
            "games_done": cfg["games"],
            "avg_pflip": None,
            "replay_freshness": None,
            "policy_entropy": None,
            "value_std": None,
            "runtime_tune": None,
        }

        # ── Self-play (Rust+NN: sole training engine) ──
        n_new = 0
        if args.concurrent:
            prev_bg = getattr(bg_worker, "_prev_count", 0)
            n_new, bg_now = wait_for_worker_progress(
                bg_worker, prev_bg, min_new=1, timeout_s=30.0)
            bg_worker._prev_count = bg_now
        elif rust_ok:
            # Rust MCTS (TT+VL+PW+QUARTZ) + batched Python NN evaluation
            states, policies, outcomes, traces = selfplay_rust_nn_batched(
                cfg, actor_source, device, cfg['games'], args.rust_binary,
                parallel=cfg.get('selfplay_parallel', 4))
            for gs, gp, out in zip(states, policies, outcomes):
                replay.add_game(gs, gp, out)
                n_new += len(gs)
            # Aggregate controller trace for logging
            all_pflips = [t.get('p_flip',0) for tr in traces for t in tr if t]
            avg_pflip = sum(all_pflips)/max(len(all_pflips),1) if all_pflips else 0
        else:
            print("ERROR: Rust binary required for training. Run: cargo build --release")
            print(f"  Expected: {args.rust_binary}")
            sys.exit(1)

        # ── Training ──
        if len(replay) >= cfg['batch']:
            train_steps = compute_train_steps(
                cfg['steps'], cfg['batch'], n_new, concurrent=args.concurrent)
            if train_steps <= 0:
                elapsed = time.time() - t0
                entry.update({
                    "replay": len(replay),
                    "new_pos": n_new,
                    "time_s": round(elapsed, 1),
                    "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
                })
                print(f"[{iteration+1:>3}/{args.iterations}] waiting for self-play: replay={len(replay)} +0 {elapsed:.1f}s")
            else:
                avg_loss, avg_pl, avg_vl, executed_steps, inner_stop = train_epoch(
                    model, optimizer, replay, cfg, device, train_steps, backend=backend,
                    inner_stop_cfg=inner_stop_cfg)
                elapsed = time.time() - t0
                if outer_stopper:
                    should_early_stop = outer_stopper.step(avg_loss)

                entry.update({
                    "loss": round(avg_loss, 4),
                    "p_loss": round(avg_pl, 4),
                    "v_loss": round(avg_vl, 4),
                    "loss_ema": round_or_none(outer_stopper.loss_ema if outer_stopper else None),
                    "replay": len(replay),
                    "new_pos": n_new,
                    "train_steps": executed_steps,
                    "planned_train_steps": train_steps,
                    "time_s": round(elapsed, 1),
                    "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
                    "avg_pflip": round(avg_pflip, 4) if avg_pflip is not None else None,
                    "replay_freshness": round(ReplayMetrics.freshness(n_new, len(replay)), 4),
                    "policy_entropy": round(ReplayMetrics.policy_entropy(replay), 3),
                    "value_std": round(ReplayMetrics.value_std(replay), 4),
                })
                if inner_stop is not None:
                    entry["inner_stop"] = inner_stop

                print(f"[{iteration+1:>3}/{args.iterations}] loss={avg_loss:.4f} (p={avg_pl:.4f} v={avg_vl:.4f}) "
                      f"lr={lr:.5f} replay={len(replay)} +{n_new} steps={executed_steps}/{train_steps} {elapsed:.1f}s")
        else:
            elapsed = time.time() - t0
            entry.update({
                "replay": len(replay),
                "new_pos": n_new,
                "time_s": round(elapsed, 1),
                "pos_per_s": round(n_new / max(elapsed, 0.1), 1),
            })
            print(f"[{iteration+1:>3}/{args.iterations}] filling replay: {len(replay)}/{cfg['batch']} +{n_new} {elapsed:.1f}s")

        if online_tuner is not None:
            runtime_overrides = online_tuner.observe(
                iteration, n_new, time.time() - t0,
                entry.get("train_steps") or 0, len(replay), bg_worker)
            if runtime_overrides:
                entry["runtime_tune"] = dict(runtime_overrides)
                changes = ", ".join(f"{k}={v}" for k, v in runtime_overrides.items())
                print(f"  [AutoTune] iter {iteration+1}: adjusted {changes}")

        # ── Checkpoint ──
        if (iteration + 1) % 5 == 0:
            if backend: backend.save(latest_model_path)
            else: torch.save(model.state_dict(), latest_model_path)
            replay.save(replay_path)
            print(f"  Checkpoint: {latest_model_path} (replay={len(replay)})")
            # Refresh background worker's model snapshot
            if bg_worker: bg_worker.update_model(actor_source)

        # ── Evaluation (Glicko-2 + Promotion) ──
        if training_evaluator and (iteration + 1) % args.eval_interval == 0 and game_factory:
            print(f"  Evaluating gen_{iteration+1} vs champion...")
            bg_pause_requested = False
            bg_was_paused = False
            cand_eng = None
            champ_eng = None
            if bg_worker and args.eval_selfplay_isolation:
                bg_pause_requested = True
                bg_was_paused = bg_worker.pause(wait=True)
            try:
                # Create engines using same Rust+NN stack as training
                candidate_name = f"gen_{iteration+1}"
                candidate_actor_template = clone_actor_model(actor_source)
                if rust_ok:
                    cand_factory = lambda: RustNNEvaluatorEngine(
                        candidate_name, cfg, clone_actor_model(candidate_actor_template), device, args.rust_binary)
                else:
                    print("  [WARN] Rust binary not found, using TreeMCTS for evaluation (NOT benchmark-grade)")
                    cand_factory = lambda: TreeMCTSEngine(candidate_name, cfg, clone_actor_model(candidate_actor_template), device)
                cand_eng = cand_factory()
                # Create champion engine from best model
                champion_actor = clone_actor_model(actor_source)
                if os.path.exists(best_model_path):
                    champion_actor = load_actor_source_from_checkpoint(
                        best_model_path, cfg, device,
                        backend_preference=backend.name if backend is not None else "torch",
                        backend_template=backend)
                champion_actor_template = clone_actor_model(champion_actor)
                if rust_ok:
                    champ_factory = lambda: RustNNEvaluatorEngine(
                        "champion", cfg, clone_actor_model(champion_actor_template), device, args.rust_binary)
                else:
                    champ_factory = lambda: TreeMCTSEngine("champion", cfg, clone_actor_model(champion_actor_template), device)
                champ_eng = champ_factory()
                if not eval_workers_autotuned and not (
                    hasattr(cand_eng, "select_moves_batch") and hasattr(champ_eng, "select_moves_batch")
                ):
                    tuned_workers, eval_benchmarks = benchmark_eval_parallel_workers(
                        hw, cfg, args.eval_games, cand_factory, champ_factory, game_factory,
                        eval_autotune_profile_path)
                    training_evaluator.cfg.parallel_workers = tuned_workers
                    eval_workers_autotuned = True
                    print(
                        f"  [EvalAutoTune] workers={tuned_workers} "
                        f"({len([b for b in eval_benchmarks if 'games_per_s' in b])} candidates benchmarked)"
                    )
                    entry["eval_worker_tune"] = {
                        "workers": tuned_workers,
                        "benchmarks": eval_benchmarks,
                    }
                elif not eval_workers_autotuned:
                    training_evaluator.cfg.parallel_workers = 1
                    eval_workers_autotuned = True
                    entry["eval_worker_tune"] = {
                        "mode": "batched_rust",
                        "workers": 1,
                        "benchmarks": [],
                    }
                    print("  [EvalAutoTune] batched Rust evaluation active (worker autotune skipped)")
                eval_result = training_evaluator.evaluate_checkpoint(
                    candidate=cand_eng, champion=champ_eng, game_factory=game_factory,
                    candidate_id=candidate_name, generation=iteration+1,
                    candidate_factory=cand_factory, champion_factory=champ_factory)
            finally:
                if bg_worker and args.eval_selfplay_isolation and bg_pause_requested:
                    bg_worker.resume()
                    print("  [BG] Self-play resumed after evaluation")
                if cand_eng is not None:
                    try:
                        cand_eng.reset()
                    except Exception:
                        pass
                if champ_eng is not None:
                    try:
                        champ_eng.reset()
                    except Exception:
                        pass
            # Log result
            eval_valid = bool(getattr(eval_result, "valid_eval", True))
            eval_invalid_reason = getattr(eval_result, "invalid_reason", None)
            v = eval_result.promotion.get("verdict", "?")
            sr = eval_result.tally.get("score_rate", 0) if eval_result.tally else 0
            elo_d = eval_result.elo.get("delta", 0) if eval_result.elo else 0
            pub = eval_result.published.get("candidate_abs") if eval_result.published else None
            champ_pub = eval_result.published.get("champion_abs") if eval_result.published else None
            elo_gap = eval_result.published.get("delta") if eval_result.published else None
            if eval_valid:
                latest_eval.update({
                    "published_elo": pub,
                    "champion_elo": champ_pub,
                    "elo_gap": elo_gap,
                    "delta_elo": elo_d,
                    "score_rate": sr,
                    "eval_verdict": v,
                })
                print(f"  Eval: {v} | sr={sr:.3f} | ΔElo={elo_d:+.0f} | AbsElo={pub} | ChampElo={champ_pub}")
                # If promoted, update best model
                if v == "promote":
                    if backend: backend.save(best_model_path)
                    else: torch.save(model.state_dict(), best_model_path)
                    print(f"  ★ PROMOTED: gen_{iteration+1} is new champion!")
            else:
                entry["eval_invalid_reason"] = str(eval_invalid_reason or "invalid evaluation")
                print(f"  [EvalInvalid] {entry['eval_invalid_reason']}")
            # Write to training log
            log_f.write(json.dumps(make_json_safe({
                                     "_type": "eval", "iter": iteration+1,
                                     "valid_eval": eval_valid,
                                     "invalid_reason": eval_invalid_reason,
                                     "verdict": v, "score_rate": sr,
                                     "delta_elo": elo_d, "published_elo": pub,
                                     "champion_elo": champ_pub, "elo_gap": elo_gap,
                                     "games": eval_result.tally.get("scored",0) if eval_result.tally else 0,
                                     "errors": eval_result.tally.get("errors",0) if eval_result.tally else 0,
                                     "voids": eval_result.tally.get("voids",0) if eval_result.tally else 0,
                                     })) + "\n")
            log_f.flush()

        entry.update({
            "published_elo": latest_eval["published_elo"],
            "champion_elo": latest_eval["champion_elo"],
            "elo_gap": latest_eval["elo_gap"],
            "delta_elo": latest_eval["delta_elo"],
            "score_rate": latest_eval["score_rate"],
            "eval_verdict": latest_eval["eval_verdict"],
        })
        log_f.write(json.dumps(make_json_safe(entry)) + "\n"); log_f.flush()

        if should_early_stop:
            print(f"\n  Early stopping at iter {iteration+1} (patience={args.patience})")
            break

    log_f.close()
    if bg_worker: bg_worker.stop()
    if backend: backend.save(latest_model_path)
    else: torch.save(model.state_dict(), latest_model_path)
    if generate_training_plots(log_path, base_dir):
        print(f"  Plots: {os.path.join(base_dir, 'training_loss.png')}"
              f" | {os.path.join(base_dir, 'training_elo.png')}")
    print(f"\nDone. Model: {latest_model_path}")

if __name__ == "__main__":
    main()
