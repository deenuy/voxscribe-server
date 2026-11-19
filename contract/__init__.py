"""Engine contract: exported schema files and the exporter that writes them.

Run the exporter:

    uv run python -m contract.export

Output files (written to this directory and checked in):

    openapi.json      OpenAPI 3.1 document for all HTTP routes
    events.json       event schema from event_schema.json_schema()
    websocket.json    machine-readable description of the live meeting WebSocket
    errors.json       named error codes with their HTTP statuses

Output is deterministic: sorted keys, no timestamps, so a diff indicates a real change.
"""
