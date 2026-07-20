"""Run the gateway: ``python -m inference_lab.gateway --config configs/gateway.example.json``."""

import argparse

import uvicorn

from inference_lab.gateway.app import create_app
from inference_lab.gateway.config import GatewayConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible routing gateway")
    parser.add_argument("--config", required=True, help="path to a GatewayConfig JSON file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config = GatewayConfig.from_file(args.config)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
