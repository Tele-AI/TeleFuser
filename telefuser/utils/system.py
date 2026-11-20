"""System utilities for port allocation and network operations."""

from __future__ import annotations

import random
import socket
import threading


class PortAllocator:
    """Thread-safe port allocator for managing non-overlapping port interval allocation."""

    _port_intervals: dict[int, tuple[int, int]] = {}
    _allocated_intervals: set[tuple[int, int]] = set()
    _lock = threading.RLock()  # Use reentrant lock to allow the same thread to acquire multiple times

    @classmethod
    def get_free_port_in_interval(
        cls, start_port: int = 10000, end_port: int = 20000, interval_length: int = 100
    ) -> int:
        """Get a free port within an unused interval (thread-safe).

        Args:
            start_port: Port range start
            end_port: Port range end
            interval_length: Length of each interval

        Returns:
            int: Available port number

        Raises:
            ValueError: When no available intervals are found
        """
        with cls._lock:
            # Calculate total number of intervals
            total_ports = end_port - start_port + 1
            num_intervals = total_ports // interval_length

            if num_intervals <= 0:
                raise ValueError("Interval length too large or port range too small")

            # Try to find unused intervals
            available_intervals = []
            for i in range(num_intervals):
                interval_start = start_port + i * interval_length
                interval_end = min(interval_start + interval_length - 1, end_port)
                interval = (interval_start, interval_end)

                if interval not in cls._allocated_intervals:
                    available_intervals.append(interval)

            if not available_intervals:
                raise ValueError("No available port intervals")

            # Randomly select an available interval
            chosen_interval = random.choice(available_intervals)
            cls._allocated_intervals.add(chosen_interval)

            # Randomly select a port within the chosen interval
            port = random.randint(chosen_interval[0], chosen_interval[1])
            cls._port_intervals[port] = chosen_interval
            return port

    @classmethod
    def release_port(cls, port: int) -> None:
        """Return the interval owned by a port from get_free_port_in_interval."""
        with cls._lock:
            interval = cls._port_intervals.pop(port, None)
            if interval is not None:
                cls._allocated_intervals.discard(interval)

    @classmethod
    def release_interval(cls, start_port: int, end_port: int):
        """Release an interval to make it available for reuse (thread-safe).

        Args:
            start_port: Interval start port
            end_port: Interval end port
        """
        with cls._lock:
            interval = (start_port, end_port)
            if interval in cls._allocated_intervals:
                cls._allocated_intervals.remove(interval)
                cls._port_intervals = {p: v for p, v in cls._port_intervals.items() if v != interval}

    @classmethod
    def get_specific_interval(cls, start_port: int, end_port: int) -> int:
        """Get a specific interval if it's available (thread-safe).

        Args:
            start_port: Interval start port
            end_port: Interval end port

        Returns:
            int: Random port within the interval

        Raises:
            ValueError: When the interval is already occupied
        """
        with cls._lock:
            interval = (start_port, end_port)
            if interval in cls._allocated_intervals:
                raise ValueError(f"Interval {start_port}-{end_port} is already occupied")

            cls._allocated_intervals.add(interval)
            return random.randint(start_port, end_port)

    @classmethod
    def reset_allocator(cls):
        """Reset the allocator, releasing all allocated intervals (thread-safe)."""
        with cls._lock:
            cls._allocated_intervals.clear()
            cls._port_intervals.clear()

    @classmethod
    def get_allocated_intervals(cls) -> set[tuple[int, int]]:
        """Get all allocated intervals (thread-safe)."""
        with cls._lock:
            return cls._allocated_intervals.copy()

    @classmethod
    def try_get_free_port_in_interval(
        cls,
        start_port: int = 30000,
        end_port: int = 40000,
        interval_length: int = 1000,
        timeout: float | None = None,
    ) -> int | None:
        """Try to get a port, can wait or return None if temporarily unavailable (thread-safe).

        Args:
            start_port: Port range start
            end_port: Port range end
            interval_length: Length of each interval
            timeout: Timeout in seconds, None means infinite wait

        Returns:
            int: Available port number, returns None on timeout
        """
        import time

        start_time = time.time()
        while True:
            # Try to acquire lock without blocking
            if cls._lock.acquire(blocking=False):
                try:
                    # Calculate available intervals
                    total_ports = end_port - start_port + 1
                    num_intervals = total_ports // interval_length

                    if num_intervals <= 0:
                        return None

                    available_intervals = []
                    for i in range(num_intervals):
                        interval_start = start_port + i * interval_length
                        interval_end = min(interval_start + interval_length - 1, end_port)
                        interval = (interval_start, interval_end)

                        if interval not in cls._allocated_intervals:
                            available_intervals.append(interval)

                    if available_intervals:
                        chosen_interval = random.choice(available_intervals)
                        cls._allocated_intervals.add(chosen_interval)
                        return random.randint(chosen_interval[0], chosen_interval[1])
                finally:
                    cls._lock.release()

            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                return None

            # Retry after brief sleep
            time.sleep(0.01)


def find_available_port(start_port: int, host: str = "localhost", max_attempts: int = 100) -> int | None:
    """Find an available port number.

    Parameters:
        start_port: Starting port number
        host: Host address, defaults to 'localhost'
        max_attempts: Maximum number of attempts, defaults to 100

    Returns:
        Available port number, returns None if not found
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port, host):
            return port
    return None


def is_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a specified port is available.

    Parameters:
        port: Port number to check
        host: Host address, defaults to 'localhost'

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def get_available_port(input_port: int | None = None, host: str = "localhost") -> int:
    """Get an available port number.

    Parameters:
        input_port: Input port number, if None or unavailable, automatically finds available port
        host: Host address, defaults to 'localhost'

    Returns:
        Available port number
    """
    # If port is specified and available, return directly
    if input_port is not None and is_port_available(input_port, host):
        return input_port

    # Otherwise find available port
    if input_port is None:
        start_port = 8000  # Default starting port
    else:
        start_port = input_port

    available_port = find_available_port(start_port, host)

    if available_port is None:
        # If no available port found, raise exception
        raise RuntimeError(f"No available port found in range {start_port} to {start_port + 100}")

    return available_port
