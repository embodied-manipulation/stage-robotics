"""Reproducible single-GPU Xiaomi-Robotics-1 inference server.

The wire protocol and model invocation match the upstream deploy/server.py.  The
only intentional additions are explicit RNG seeds, inference_mode, and a clean
one-client shutdown mode for a foreground Cluster Worker.
"""

from __future__ import annotations

import argparse
import pickle
import random
import socket
import struct
import traceback

import numpy as np
import torch
from transformers import AutoModel


def recv_all(connection: socket.socket, length: int) -> bytes | None:
    data = bytearray()
    while len(data) < length:
        packet = connection.recv(length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load-device", choices=("cpu", "cuda"), default="cpu",
                        help="Load checkpoint shards directly to CUDA to limit host memory")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Sequential episode clients sharing one loaded checkpoint")
    parser.add_argument(
        "--attention",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(
        f"Loading XR-1 model={args.model} attention={args.attention} seed={args.seed}",
        flush=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        attn_implementation=args.attention,
        dtype=torch.bfloat16,
        **({'device_map': 'cuda', 'low_cpu_mem_usage': True} if args.load_device == 'cuda' else {}),
    ).cuda().to(torch.bfloat16)
    model.eval()
    print("XR-1 model loaded", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((args.host, args.port))
        server_socket.listen(1)
        print(f"READY {args.host}:{args.port}", flush=True)
        request_count = 0
        for session in range(args.sessions):
            connection, address = server_socket.accept()
            print(f"Client connected: {address} episode={session} checkpoint_loads=1", flush=True)
            # Reset episode randomness without reloading the frozen model.
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            try:
                with connection:
                    while True:
                        length_bytes = recv_all(connection, 4)
                        if not length_bytes:
                            break
                        data_length = struct.unpack(">I", length_bytes)[0]
                        payload = recv_all(connection, data_length)
                        if payload is None:
                            break
                        input_data = pickle.loads(payload)
                        device_data = {
                            key: (
                                value.to(device=model.device, dtype=model.dtype)
                                if isinstance(value, torch.Tensor) and value.is_floating_point()
                                else value.to(device=model.device)
                                if isinstance(value, torch.Tensor) else value
                            ) for key, value in input_data.items()
                        }
                        with torch.inference_mode():
                            outputs = model(**device_data)
                        response = pickle.dumps(outputs.actions.cpu(), protocol=pickle.HIGHEST_PROTOCOL)
                        connection.sendall(struct.pack(">I", len(response)) + response)
                        request_count += 1
                        if request_count % 25 == 0:
                            print(f"requests={request_count}", flush=True)
            except Exception:
                traceback.print_exc()
                raise
    print(f"Server stopped after {request_count} requests", flush=True)


if __name__ == "__main__":
    main()
