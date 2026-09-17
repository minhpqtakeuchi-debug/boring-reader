import asyncio
import json
from pathlib import Path

from mcp.client.stdio import StdioTransport
from mcp.client.session import ClientSession

PDF_PATH = "root/boring (29).pdf"  # adjust to an existing file


async def main():
    # 0) Read the PDF as bytes (your MCP tool expects pdf_bytes)
    pdf_path = Path(PDF_PATH)
    with pdf_path.open("rb") as f:
        pdf_bytes = f.read()

    # 1) Launch the MCP server over stdio
    transport = StdioTransport(
        command=["python", "mcp/boring_reader_server.py"],
    )

    async with transport:
        session = ClientSession(transport)

        # 2) Handshake / initialize
        await session.initialize()

        # 3) List tools to verify server is alive
        tools = await session.list_tools()
        print("Available tools:", [t.name for t in tools.tools])

        # 4) Call create_boring_batch
        create_args = {
            "pdf_bytes": pdf_bytes,        # <-- bytes, not path
            "max_output_tokens_g3": 10_000,
        }

        create_resp = await session.call_tool(
            "create_boring_batch",
            arguments=create_args,
        )

        # FastMCP will wrap your dict result as JSON content.
        # Grab the first JSON-ish content block.
        job_info = None
        for c in create_resp.content:
            # Depending on mcp version this may be .json or .data
            if hasattr(c, "json") and c.json is not None:
                job_info = c.json
                break
            if hasattr(c, "data") and isinstance(c.data, (dict, list)):
                job_info = c.data
                break

        print("create_boring_batch raw content objects:", create_resp.content)
        print("Parsed job_info:", json.dumps(job_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
