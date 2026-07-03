"""Helpers for constraining WebRTC ICE host candidate gathering."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Iterable


def _normalize_host_ips(host_ips: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_host_ip in host_ips:
        host_ip = raw_host_ip.strip()
        if not host_ip:
            continue
        canonical_host_ip = str(ipaddress.ip_address(host_ip))
        if canonical_host_ip in seen:
            continue
        seen.add(canonical_host_ip)
        normalized.append(canonical_host_ip)
    return normalized


def _matches_ip_version(host_ip: str, *, use_ipv4: bool, use_ipv6: bool) -> bool:
    version = ipaddress.ip_address(host_ip).version
    if version == 4:
        return use_ipv4
    if version == 6:
        return use_ipv6
    return False


def configure_ice_host_addresses(host_ips: Iterable[str] | None = None) -> None:
    """Limit aiortc host candidate gathering to a small, explicit IP allowlist."""

    if host_ips is None:
        raw_host_ips = os.environ.get("TELEFUSER_WEBRTC_ICE_HOST_IPS", "")
        if not raw_host_ips.strip():
            return
        host_ips = raw_host_ips.split(",")

    allowed_host_ips = _normalize_host_ips(host_ips)
    if not allowed_host_ips:
        return

    try:
        from aioice import ice as aioice_ice
    except Exception:
        return

    current_get_host_addresses = aioice_ice.get_host_addresses
    if getattr(current_get_host_addresses, "_telefuser_host_filter", False):
        return

    def _get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
        filtered = [
            host_ip
            for host_ip in allowed_host_ips
            if _matches_ip_version(host_ip, use_ipv4=use_ipv4, use_ipv6=use_ipv6)
        ]
        if filtered:
            return filtered
        return current_get_host_addresses(use_ipv4=use_ipv4, use_ipv6=use_ipv6)

    _get_host_addresses._telefuser_host_filter = True  # type: ignore[attr-defined]
    aioice_ice.get_host_addresses = _get_host_addresses
