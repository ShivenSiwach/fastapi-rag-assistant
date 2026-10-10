from mcp.server import MCPServer

# The server object. The name is what clients will display.
mcp = MCPServer("hello-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two whole numbers and return the sum."""
    return a + b


@mcp.tool()
def greet(name: str, formal: bool = False) -> str:
    """Greet a person by name. Set formal=True for a formal greeting."""
    if formal:
        return f"Good day, {name}. It is a pleasure to meet you."
    return f"Hey {name}!"

@mcp.tool()
def noisy(text: str) -> str:
    """Print a debug line to stdout, then return the text in uppercase."""
    print("DEBUG: stray print to stdout")
    return text.upper()


if __name__ == "__main__":
    # With no arguments this starts a stdio server: it waits for a
    # client to talk to it through stdin/stdout.
    mcp.run()