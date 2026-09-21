"""Length-prefixed message framing for the Isaac Sim <-> planner socket.

A 640x480 float32 depth frame is 1.2 MB, which JSON cannot carry sensibly, so
each message is a small JSON header followed by an optional raw binary payload:

    [4 bytes: header length, big-endian][header JSON][payload bytes]

The header's "nbytes" field says how long the payload is; omit it for
header-only messages.
"""

import json
import socket
import struct

_HDR = struct.Struct(">I")


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    """Send one framed message."""
    if payload:
        header = {**header, "nbytes": len(payload)}
    raw = json.dumps(header).encode("utf-8")
    sock.sendall(_HDR.pack(len(raw)) + raw + payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, or raise if the peer closes first."""
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(1 << 20, n - got))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket):
    """Receive one framed message as (header dict, payload bytes)."""
    (size,) = _HDR.unpack(_recv_exactly(sock, _HDR.size))
    header = json.loads(_recv_exactly(sock, size).decode("utf-8"))
    payload = _recv_exactly(sock, header["nbytes"]) if header.get("nbytes") else b""
    return header, payload
