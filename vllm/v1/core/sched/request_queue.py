# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request

import time
import threading
import queue as queue_module

from vllm.v1.core.sched.ua_predictor import predict_length_from_token_ids as ua_predict, predict_length_from_token_ids_batch as ua_predict_batch

NOT_FOUND_COUNT = 0

# Starvation prevention constants (paper §5): Score' = Score · γ^(tw/τ)
STARVATION_GAMMA = 0.9    # decay factor γ
STARVATION_TAU   = 30.0   # decay interval τ (seconds)
STARVATION_UPDATE_INTERVAL = 5.0  # period between heap re-heapifies (seconds)


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    SSJF = "ssjf"
    UA = "ua"
    LTR = "ltr"
    GMM_UA = "gmm_ua"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [
            (p, t, r) for p, t, r in self._heap if r not in requests_to_remove
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


class UARequestQueue(RequestQueue):
    """Min-heap waiting queue for the TIE scheduler.

    Requests are sorted by their TIE score (E[X] + β·CVaR), predicted
    asynchronously by a background thread. Unpredicted requests are placed
    at the bottom of the heap (initial score = max_tokens). A starvation
    prevention mechanism periodically decays scores so long-waiting requests
    gradually rise in priority.

    Uses lazy deletion to achieve the complexities stated in the paper:
      peek  : O(1) amortized   (skips stale entries at the top)
      push  : O(log n)
      pop   : O(log n) amortized
      update: O(log n)         (push new entry; old entry becomes stale)
      remove: O(1)             (invalidate version; stale entry cleaned lazily)

    Each heap entry is a 5-tuple (effective_score, arrival_time, version,
    req_id, request). A version counter per request identifies which heap
    entry is current; entries with an outdated version are discarded on
    pop/peek without affecting correctness.

    The periodic starvation-decay rebuild (_apply_starvation_decay) is O(n)
    but runs in the background every STARVATION_UPDATE_INTERVAL seconds and
    is not part of the per-request scheduling path.
    """

    _ua_predict = None

    def __init__(
        self,
        tokenizer=None,
        max_batch_size: int = 128,
        optimal_batch_size: int = 8,
        max_wait_time_ms: float = 3.0,
        enable_batching: bool = True
    ) -> None:
        # Heap entries: (effective_score, arrival_time, version, req_id, request)
        self._heap: list[tuple[float, float, int, str, Request]] = []
        self._lock = threading.RLock()
        self.tokenizer = tokenizer

        # Lazy-deletion bookkeeping (all keyed by request_id)
        self._versions: dict[str, int] = {}       # current valid heap entry version
        self._base_scores: dict[str, float] = {}  # un-decayed TIE score
        # O(1) lookup of (arrival_time, request) needed by _push_updated_score
        self._request_info: dict[str, tuple[float, Request]] = {}

        self._prediction_queue = queue_module.Queue()
        self._running = True

        self.max_batch_size = max_batch_size
        self.optimal_batch_size = optimal_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_batching = enable_batching

        print("[UA] Loading predictor...")
        print(f"[UA] Batch config: max_batch={max_batch_size}, "
              f"optimal_batch={optimal_batch_size}, "
              f"max_wait={max_wait_time_ms}ms, "
              f"batching={enable_batching}")
        try:
            UARequestQueue._ua_predict = ua_predict
            UARequestQueue._ua_predict_batch = ua_predict_batch
        except Exception as e:
            print(f"[UA] Failed to load predictor: {e}. Falling back to max_tokens.")
            UARequestQueue._ua_predict = None

        self._prediction_thread = threading.Thread(
            target=self._prediction_worker,
            daemon=True,
            name="UA-Predictor"
        )
        self._prediction_thread.start()
        print("[UA] Background prediction thread started.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_updated_score(self, req_id: str, new_base: float) -> None:
        """Push a new heap entry for req_id with an updated score (O(log n)).

        Increments the version counter so the previous heap entry becomes
        stale and will be discarded on the next pop/peek.

        Must be called with self._lock held.
        """
        if req_id not in self._versions:
            return  # request was already popped or removed
        arrival_time, request = self._request_info[req_id]
        current_time = time.time()
        tw = max(0.0, current_time - arrival_time)
        effective = new_base * (STARVATION_GAMMA ** (tw / STARVATION_TAU))
        new_version = self._versions[req_id] + 1
        self._versions[req_id] = new_version
        self._base_scores[req_id] = new_base
        heapq.heappush(
            self._heap,
            (effective, arrival_time, new_version, req_id, request)
        )

    def _apply_starvation_decay(self) -> None:
        """Re-heapify with time-decayed scores to prevent request starvation.

        Applies Score' = Score · γ^(tw/τ) (paper §5, γ=0.9, τ=30 s).
        Long-waiting requests receive lower effective scores and thus rise
        toward the top of the min-heap.

        This is a global O(n) rebuild. It runs in the background every
        STARVATION_UPDATE_INTERVAL seconds and is not part of the per-request
        O(log n) scheduling path. All versions are bumped so stale entries
        accumulated by lazy deletion are purged in the same pass.
        """
        with self._lock:
            if not self._versions:
                return
            current_time = time.time()
            new_heap = []
            for req_id, (arrival_time, request) in self._request_info.items():
                base = self._base_scores.get(req_id, 2048.0)
                tw = max(0.0, current_time - arrival_time)
                effective = base * (STARVATION_GAMMA ** (tw / STARVATION_TAU))
                new_version = self._versions[req_id] + 1
                self._versions[req_id] = new_version
                new_heap.append(
                    (effective, arrival_time, new_version, req_id, request)
                )
            self._heap = new_heap
            heapq.heapify(self._heap)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _prediction_worker(self) -> None:
        """Background thread: process prediction tasks and apply starvation decay."""
        batch_buffer = []
        last_process_time = time.time()
        last_starvation_update = time.time()
        while self._running:
            # Periodically apply starvation decay (paper §5)
            current_time = time.time()
            if current_time - last_starvation_update >= STARVATION_UPDATE_INTERVAL:
                self._apply_starvation_decay()
                last_starvation_update = current_time
            try:
                task = self._prediction_queue.get(timeout=0.005)
                if task is None:
                    break
                batch_buffer.append(task)
                elapsed_ms = (time.time() - last_process_time) * 1000
                queue_size = self._prediction_queue.qsize()

                should_process = (
                    len(batch_buffer) >= self.max_batch_size or
                    (len(batch_buffer) >= self.optimal_batch_size and queue_size == 0) or
                    elapsed_ms >= self.max_wait_time_ms or
                    not self.enable_batching
                )

                if should_process:
                    self._process_batch(batch_buffer)
                    batch_buffer = []
                    last_process_time = time.time()
            except queue_module.Empty:
                if batch_buffer:
                    self._process_batch(batch_buffer)
                    batch_buffer = []
                    last_process_time = time.time()
                continue
            except Exception as e:
                print(f"[UA] Prediction thread error: {e}")
                import traceback
                traceback.print_exc()
                batch_buffer = []

    def _batch_update_service_times(self, updates: list[tuple[str, int]]) -> None:
        """Batch-update TIE scores via lazy deletion: O(k log n) for k updates.

        Args:
            updates: list of (request_id, new_tie_score) pairs.
        """
        global NOT_FOUND_COUNT

        if not updates:
            return

        with self._lock:
            not_found = 0
            for req_id, new_score in updates:
                if req_id in self._versions:
                    self._push_updated_score(req_id, float(new_score))
                else:
                    not_found += 1
            if not_found > 0:
                NOT_FOUND_COUNT += not_found

    def _process_batch(self, batch_buffer: list) -> None:
        """Run the predictor on a batch and update the heap."""
        if not batch_buffer:
            return

        if not self.tokenizer or not UARequestQueue._ua_predict:
            return

        try:
            requests = [task[0] for task in batch_buffer]

            valid_indices = []
            token_ids_list = []
            for i, request in enumerate(requests):
                if request.prompt_token_ids:
                    valid_indices.append(i)
                    token_ids_list.append(request.prompt_token_ids)

            if not valid_indices:
                return

            with self._lock:
                waiting_count = len(self._versions)

            if (self.enable_batching and
                len(valid_indices) > 1 and
                UARequestQueue._ua_predict_batch is not None):

                start_time = time.time()
                predicted_outputs = UARequestQueue._ua_predict_batch(
                    token_ids_list,
                    self.tokenizer,
                    waiting_count
                )
                elapsed = time.time() - start_time
                print(f"[UA] Batch prediction done: size={len(valid_indices)}, "
                      f"total={elapsed*1000:.1f}ms, "
                      f"avg={elapsed*1000/len(valid_indices):.1f}ms/req")

                updates = []
                for idx, predicted_output in zip(valid_indices, predicted_outputs):
                    request = requests[idx]
                    updates.append((request.request_id, predicted_output))
                self._batch_update_service_times(updates)
            else:
                for idx in valid_indices:
                    request = requests[idx]
                    try:
                        predicted_output = UARequestQueue._ua_predict(
                            request.prompt_token_ids,
                            self.tokenizer,
                            waiting_count
                        )
                        self._update_service_time(request.request_id, predicted_output)
                    except Exception as e:
                        print(f"[UA] Single prediction failed: {e}")

        except Exception as e:
            print(f"[UA] Batch processing failed: {e}")
            import traceback
            traceback.print_exc()

            for request, _ in batch_buffer:
                try:
                    if request.prompt_token_ids and UARequestQueue._ua_predict:
                        with self._lock:
                            waiting_count = len(self._versions)
                        predicted_output = UARequestQueue._ua_predict(
                            request.prompt_token_ids,
                            self.tokenizer,
                            waiting_count
                        )
                        self._update_service_time(request.request_id, predicted_output)
                except:
                    pass

    def _update_service_time(self, request_id: str, new_service_time: int) -> None:
        """Update the TIE score of a single request: O(log n) via lazy deletion."""
        global NOT_FOUND_COUNT
        with self._lock:
            if request_id in self._versions:
                self._push_updated_score(request_id, float(new_service_time))
            else:
                NOT_FOUND_COUNT += 1

    # ------------------------------------------------------------------
    # Public queue interface
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Enqueue a request with max_tokens as its initial score (O(log n)),
        then submit an asynchronous prediction task to refine the score."""
        initial_score = 2048.0
        req_id = request.request_id

        with self._lock:
            self._versions[req_id] = 0
            self._base_scores[req_id] = initial_score
            self._request_info[req_id] = (request.arrival_time, request)
            heapq.heappush(
                self._heap,
                (initial_score, request.arrival_time, 0, req_id, request)
            )

        if UARequestQueue._ua_predict is not None and self.tokenizer:
            self._prediction_queue.put((request, initial_score))

    def pop_request(self) -> Request:
        """Remove and return the request with the lowest effective score (O(log n) amortized).

        Stale heap entries (created by lazy score updates or removals) are
        discarded until a live entry is found.
        """
        with self._lock:
            while self._heap:
                _, _, version, req_id, request = heapq.heappop(self._heap)
                if self._versions.get(req_id) == version:
                    del self._versions[req_id]
                    del self._base_scores[req_id]
                    del self._request_info[req_id]
                    return request
            raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Return (without removing) the request with the lowest effective score (O(1) amortized).

        Stale entries at the heap top are discarded until a live entry is found.
        """
        with self._lock:
            while self._heap:
                _, _, version, req_id, request = self._heap[0]
                if self._versions.get(req_id) == version:
                    return request
                heapq.heappop(self._heap)  # discard stale entry at top
            raise IndexError("peek from empty heap")

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a request from the queue (O(1)).

        Invalidates the request's version so its heap entry is discarded
        lazily on the next pop/peek.
        """
        req_id = request.request_id
        with self._lock:
            if req_id in self._versions:
                del self._versions[req_id]
                del self._base_scores[req_id]
                del self._request_info[req_id]

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple requests from the queue (O(k) for k requests)."""
        with self._lock:
            for request in requests:
                req_id = request.request_id
                if req_id in self._versions:
                    del self._versions[req_id]
                    del self._base_scores[req_id]
                    del self._request_info[req_id]

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._versions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._versions)

    def __iter__(self) -> Iterator[Request]:
        """Yield requests in ascending score order, skipping stale heap entries."""
        with self._lock:
            heap_copy = self._heap[:]
            versions_copy = dict(self._versions)
        while heap_copy:
            _, _, version, req_id, request = heapq.heappop(heap_copy)
            if versions_copy.get(req_id) == version:
                versions_copy.pop(req_id)
                yield request

    def __reversed__(self) -> Iterator[Request]:
        return reversed(list(self))

    def shutdown(self) -> None:
        """Shut down the background prediction thread."""
        print("[UA] Shutting down prediction thread...")
        self._running = False
        self._prediction_queue.put(None)

        if self._prediction_thread.is_alive():
            self._prediction_thread.join(timeout=2.0)
            if self._prediction_thread.is_alive():
                print("[UA] Warning: prediction thread did not stop cleanly.")
            else:
                print("[UA] Prediction thread stopped.")


def create_request_queue(policy: SchedulingPolicy, tokenizer=None) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.UA:
        return UARequestQueue(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
