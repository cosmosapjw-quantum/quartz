"""
Multi-Process Inference Architecture (Phase 6)
==============================================

Bypasses Python GIL entirely by running inference in separate processes.
Uses shared memory for zero-copy tensor handoff between C++ and Python.

Architecture:
    Main Process (C++)                  Inference Process 1
         ↓                                       ↓
    Simulation Threads              PyTorch Model (GPU)
         ↓                                       ↓
    AsyncInferenceQueue ─────────→  Shared Memory Tensors
         ↓                                       ↓
    Request Semaphore  ──────────→  Wakes Process 1
         ↓                                       ↓
    Result Semaphore  ←──────────  Process Completes
         ↓
    Continue Simulation

Key Design Points:
1. Inference processes start at initialization, stay alive
2. Shared memory arrays for tensors (POSIX shm)
3. Semaphores for synchronization (no polling)
4. Process pool (N=2-4 processes for multi-GPU or CPU parallelism)
5. Zero GIL contention (each process has its own GIL)

Performance Target:
- 20,000-35,000 sims/sec (3-5× improvement over single process)
- Python callback overhead <1ms (vs 5-10ms with GIL)
- Near-linear scaling with multiple GPUs

"""

import multiprocessing as mp
from multiprocessing import shared_memory, Semaphore, Process, Queue
import numpy as np
import torch
import time
import logging
import signal
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SharedTensorSlot:
    """Shared memory slot for tensor handoff."""
    features_shm: shared_memory.SharedMemory
    policies_shm: shared_memory.SharedMemory
    values_shm: shared_memory.SharedMemory

    # Metadata (sizes, counts)
    batch_size: int
    num_planes: int
    board_size: int
    action_space: int

    # Synchronization
    request_ready: Semaphore
    result_ready: Semaphore


class InferenceWorkerProcess:
    """Separate process that runs GPU inference without GIL contention.

    Each worker:
    1. Loads model into GPU memory
    2. Waits on semaphore for batches
    3. Runs inference in shared memory
    4. Signals completion

    No GIL contention because each process has its own interpreter!
    """

    def __init__(self, worker_id: int, model_factory, device: str):
        """Initialize worker (called in main process before fork)."""
        self.worker_id = worker_id
        self.model_factory = model_factory
        self.device = device
        self.logger = logging.getLogger(f"Worker-{worker_id}")

    def run(self, slot: SharedTensorSlot, shutdown_event: mp.Event):
        """Worker loop (runs in separate process)."""
        # Signal handlers for clean shutdown
        def signal_handler(signum, frame):
            self.logger.info(f"Worker {self.worker_id} received shutdown signal")
            shutdown_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Load model in this process (each process has its own GPU context)
            self.logger.info(f"Worker {self.worker_id} loading model on {self.device}...")
            model = self.model_factory()
            model = model.to(self.device)
            model.eval()
            self.logger.info(f"Worker {self.worker_id} ready")

            # Main inference loop
            while not shutdown_event.is_set():
                # Wait for request (blocks on semaphore)
                if not slot.request_ready.acquire(timeout=0.1):
                    continue

                if shutdown_event.is_set():
                    break

                try:
                    # Read features from shared memory (zero-copy!)
                    features_np = np.ndarray(
                        (slot.batch_size, slot.num_planes, slot.board_size, slot.board_size),
                        dtype=np.float32,
                        buffer=slot.features_shm.buf
                    )

                    # Convert to tensor (still zero-copy via numpy's buffer)
                    features_tensor = torch.from_numpy(features_np).to(self.device, non_blocking=True)

                    # Run inference (NO GIL - this process owns its own GIL!)
                    with torch.inference_mode():
                        policy_logits, values = model(features_tensor)
                        policies = torch.softmax(policy_logits, dim=-1)

                    # Write results to shared memory (zero-copy!)
                    policies_np = np.ndarray(
                        (slot.batch_size, slot.action_space),
                        dtype=np.float32,
                        buffer=slot.policies_shm.buf
                    )
                    values_np = np.ndarray(
                        (slot.batch_size,),
                        dtype=np.float32,
                        buffer=slot.values_shm.buf
                    )

                    policies_np[:] = policies.cpu().numpy()
                    values_np[:] = values.cpu().numpy().flatten()

                    # Signal completion
                    slot.result_ready.release()

                except Exception as e:
                    self.logger.error(f"Worker {self.worker_id} inference error: {e}")
                    slot.result_ready.release()  # Signal even on error to avoid deadlock

        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} fatal error: {e}")
        finally:
            self.logger.info(f"Worker {self.worker_id} shutting down")


class MultiProcessInferenceManager:
    """Manages pool of inference worker processes.

    Coordinates:
    - Shared memory allocation
    - Process lifecycle
    - Load balancing across workers
    - Clean shutdown

    Usage:
        manager = MultiProcessInferenceManager(
            model_factory=lambda: create_model('gomoku'),
            num_workers=2,
            device='cuda'
        )
        manager.start()

        # Use like regular callback
        policies, values = manager.batch_inference(features_batch, board_sizes, num_planes_list)

        manager.stop()
    """

    def __init__(self,
                 model_factory,
                 num_workers: int = 2,
                 device: str = 'cuda',
                 max_batch_size: int = 128):
        """Initialize manager.

        Args:
            model_factory: Callable that returns model instance
            num_workers: Number of inference processes (default: 2)
            device: Device for inference
            max_batch_size: Maximum batch size for shared memory allocation
        """
        self.model_factory = model_factory
        self.num_workers = num_workers
        self.device = device
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)

        # Worker processes and shared memory
        self.workers: List[Process] = []
        self.slots: List[SharedTensorSlot] = []
        self.shutdown_events: List[mp.Event] = []

        # Round-robin load balancing
        self._next_worker = 0

        self.running = False

    def _create_shared_slot(self, slot_id: int) -> SharedTensorSlot:
        """Create shared memory slot for one worker."""
        # Assume Gomoku for now (15×15, 36 planes, 225 actions)
        # TODO: Make this configurable
        batch_size = self.max_batch_size
        num_planes = 36
        board_size = 15
        action_space = 225

        # Allocate shared memory
        features_size = batch_size * num_planes * board_size * board_size * 4  # float32
        policies_size = batch_size * action_space * 4
        values_size = batch_size * 4

        # Create shared memory (with cleanup of any leftover from previous run)
        import os
        shm_name_prefix = f"mcts_phase6_{os.getpid()}_"

        try:
            features_shm = shared_memory.SharedMemory(create=True, size=features_size, name=f"{shm_name_prefix}features_{slot_id}")
        except FileExistsError:
            # Clean up leftover and retry
            existing = shared_memory.SharedMemory(name=f"{shm_name_prefix}features_{slot_id}")
            existing.close()
            existing.unlink()
            features_shm = shared_memory.SharedMemory(create=True, size=features_size, name=f"{shm_name_prefix}features_{slot_id}")

        try:
            policies_shm = shared_memory.SharedMemory(create=True, size=policies_size, name=f"{shm_name_prefix}policies_{slot_id}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=f"{shm_name_prefix}policies_{slot_id}")
            existing.close()
            existing.unlink()
            policies_shm = shared_memory.SharedMemory(create=True, size=policies_size, name=f"{shm_name_prefix}policies_{slot_id}")

        try:
            values_shm = shared_memory.SharedMemory(create=True, size=values_size, name=f"{shm_name_prefix}values_{slot_id}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=f"{shm_name_prefix}values_{slot_id}")
            existing.close()
            existing.unlink()
            values_shm = shared_memory.SharedMemory(create=True, size=values_size, name=f"{shm_name_prefix}values_{slot_id}")

        # Create semaphores
        request_ready = Semaphore(0)  # Initially locked
        result_ready = Semaphore(0)   # Initially locked

        return SharedTensorSlot(
            features_shm=features_shm,
            policies_shm=policies_shm,
            values_shm=values_shm,
            batch_size=batch_size,
            num_planes=num_planes,
            board_size=board_size,
            action_space=action_space,
            request_ready=request_ready,
            result_ready=result_ready
        )

    def start(self):
        """Start all worker processes."""
        if self.running:
            self.logger.warning("Manager already running")
            return

        self.logger.info(f"Starting {self.num_workers} inference worker processes...")

        # Create shared memory slots
        for i in range(self.num_workers):
            slot = self._create_shared_slot(i)
            self.slots.append(slot)

        # Create and start worker processes
        for i in range(self.num_workers):
            shutdown_event = mp.Event()
            self.shutdown_events.append(shutdown_event)

            worker = InferenceWorkerProcess(i, self.model_factory, self.device)
            process = Process(
                target=worker.run,
                args=(self.slots[i], shutdown_event),
                name=f"InferenceWorker-{i}"
            )
            process.start()
            self.workers.append(process)

            self.logger.info(f"  Worker {i} started (PID: {process.pid})")

        # Wait for workers to initialize (with verification)
        max_wait = 5.0
        start_wait = time.time()
        all_alive = False

        while time.time() - start_wait < max_wait:
            all_alive = all(p.is_alive() for p in self.workers)
            if all_alive:
                break
            time.sleep(0.1)

        if not all_alive:
            self.logger.error("Some workers failed to start!")
            for i, p in enumerate(self.workers):
                if not p.is_alive():
                    self.logger.error(f"  Worker {i}: exitcode={p.exitcode}")

        self.running = True
        self.logger.info(f"Multi-process inference manager ready ({self.num_workers} workers, all_alive={all_alive})")

    def stop(self):
        """Stop all worker processes and clean up shared memory."""
        if not self.running:
            return

        self.logger.info("Stopping inference workers...")

        # Signal shutdown
        for event in self.shutdown_events:
            event.set()

        # Terminate processes
        for i, process in enumerate(self.workers):
            process.join(timeout=5.0)
            if process.is_alive():
                self.logger.warning(f"Worker {i} did not stop gracefully, terminating...")
                process.terminate()
                process.join(timeout=2.0)

        # Clean up shared memory
        for i, slot in enumerate(self.slots):
            try:
                slot.features_shm.close()
                slot.features_shm.unlink()
                slot.policies_shm.close()
                slot.policies_shm.unlink()
                slot.values_shm.close()
                slot.values_shm.unlink()
            except Exception as e:
                self.logger.error(f"Error cleaning up slot {i}: {e}")

        self.workers.clear()
        self.slots.clear()
        self.shutdown_events.clear()

        self.running = False
        self.logger.info("Multi-process inference manager stopped")

    def batch_inference_features(self, features_batch, board_sizes, num_planes_list):
        """Batch inference using worker processes (GIL-free!).

        This is the callback interface expected by BatchInferenceCoordinator.
        """
        if not self.running:
            raise RuntimeError("Manager not running")

        batch_size = len(features_batch)
        if batch_size == 0:
            return []

        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")

        # Select worker (round-robin)
        worker_id = self._next_worker
        self._next_worker = (self._next_worker + 1) % self.num_workers
        slot = self.slots[worker_id]

        # Write features to shared memory
        features_np = np.ndarray(
            (slot.batch_size, slot.num_planes, slot.board_size, slot.board_size),
            dtype=np.float32,
            buffer=slot.features_shm.buf
        )

        for i, (features, board_size, num_planes) in enumerate(zip(features_batch, board_sizes, num_planes_list)):
            features_array = np.array(features, dtype=np.float32).reshape(num_planes, board_size, board_size)
            features_np[i, :num_planes, :board_size, :board_size] = features_array

        # Signal worker
        slot.request_ready.release()

        # Wait for result (blocks on semaphore)
        if not slot.result_ready.acquire(timeout=10.0):
            raise TimeoutError(f"Worker {worker_id} did not respond within 10 seconds")

        # Read results from shared memory
        policies_np = np.ndarray(
            (slot.batch_size, slot.action_space),
            dtype=np.float32,
            buffer=slot.policies_shm.buf
        )
        values_np = np.ndarray(
            (slot.batch_size,),
            dtype=np.float32,
            buffer=slot.values_shm.buf
        )

        # Convert to list format
        results = []
        for i in range(batch_size):
            policy = policies_np[i].tolist()
            value = float(values_np[i])
            results.append((policy, value))

        return results


def create_multiprocess_inference_callback(model_factory, num_workers=2, device='cuda'):
    """Factory function to create multi-process inference callback.

    Args:
        model_factory: Callable that returns model instance
        num_workers: Number of inference processes
        device: Device for inference

    Returns:
        mcts_py.PyBatchInferenceCallback wrapping MultiProcessInferenceManager
    """
    import mcts_py

    manager = MultiProcessInferenceManager(model_factory, num_workers, device)
    manager.start()

    # Wrap in PyBatchInferenceCallback
    callback = mcts_py.PyBatchInferenceCallback(manager.batch_inference_features)

    # Store reference to manager for cleanup
    callback._manager = manager

    return callback
