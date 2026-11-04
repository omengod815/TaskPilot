"""Optional MCP server exposing a small, explicitly bounded subset of TaskPilot tools."""
from mcp.server.fastmcp import FastMCP

from app.tools.toolkit import analyze_csv, safe_calculate, search_local_knowledge

mcp = FastMCP("taskpilot-tools")


@mcp.tool()
def calculate(expression: str) -> dict:
    """Evaluate a safe arithmetic expression. Arbitrary Python is not executed."""
    return safe_calculate(expression)


@mcp.tool()
def analyze_csv_file(filename: str) -> dict:
    """Summarize a CSV stored in the approved data/public or data/uploads directory."""
    return analyze_csv(filename)


@mcp.tool()
def search_local_notes(query: str, top_k: int = 4) -> dict:
    """Search the local Markdown demo knowledge base; this is not live web search."""
    return search_local_knowledge(query, top_k)


if __name__ == "__main__":
    mcp.run()
