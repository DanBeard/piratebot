"""
Voice Line Search Tools for LLM Integration.

Provides tool definitions and handlers for the LLM to search and select
pre-generated voice lines.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.ollama_llm import ToolDefinition, ToolCall, ToolResult
from services.voice_line_tts import VoiceLineTTS
from services.voice_line_db import VoiceLine

logger = logging.getLogger(__name__)


# Tool definitions for voice line selection
VOICE_LINE_TOOLS = [
    ToolDefinition(
        name="search_voice_lines",
        description=(
            "Search for pre-recorded pirate voice lines that match what you want to say. "
            "Returns a list of matching lines with similarity scores. Use this to find "
            "appropriate responses for trick-or-treaters based on their costume."
        ),
        parameters={
            "type": "object",
            "properties": {
                "desired_text": {
                    "type": "string",
                    "description": "What you want to say to the trick-or-treater",
                },
                "category_hint": {
                    "type": "string",
                    "description": "Category to search in",
                    "enum": [
                        "greetings",
                        "costume_reactions",
                        "farewells",
                        "idle",
                        "jokes",
                        "fallbacks",
                    ],
                },
                "costume_type": {
                    "type": "string",
                    "description": (
                        "Type of costume for better matching: vampire, zombie, witch, "
                        "wizard, superhero, princess, ghost, skeleton, mummy, werewolf, "
                        "ninja, dinosaur, cat, pirate, astronaut, doctor"
                    ),
                },
            },
            "required": ["desired_text"],
        },
    ),
    ToolDefinition(
        name="select_voice_line",
        description=(
            "Select a specific voice line by ID to play. Use this after searching "
            "to choose the best matching line. The selected line will be played "
            "with the pirate's voice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "line_id": {
                    "type": "string",
                    "description": "The ID of the voice line to play",
                },
            },
            "required": ["line_id"],
        },
    ),
    ToolDefinition(
        name="get_random_greeting",
        description=(
            "Get a random greeting when you don't need to search. "
            "Use this for simple hellos when you haven't seen the costume yet."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="get_random_farewell",
        description=(
            "Get a random farewell to say goodbye to a trick-or-treater. "
            "Use this when they're leaving."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


@dataclass
class VoiceLineSelection:
    """Result of the voice line selection process."""

    line: Optional[VoiceLine]
    text: str  # The text that will be spoken
    line_id: str
    method: str  # "search", "random", "fallback"
    search_score: float = 0.0


class VoiceLineToolHandler:
    """
    Handles tool calls from the LLM for voice line selection.

    This class processes tool calls and returns appropriate results,
    eventually selecting a voice line to play.
    """

    def __init__(self, tts: VoiceLineTTS):
        """
        Initialize tool handler.

        Args:
            tts: VoiceLineTTS instance for line lookup
        """
        self.tts = tts
        self._pending_search_results: list[dict] = []
        self._selected_line: Optional[VoiceLineSelection] = None

    def get_tools(self) -> list[ToolDefinition]:
        """Get the list of available tools."""
        return VOICE_LINE_TOOLS

    def handle_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """
        Handle a tool call from the LLM.

        Args:
            tool_call: The tool call to handle

        Returns:
            ToolResult with the response
        """
        try:
            if tool_call.name == "search_voice_lines":
                return self._handle_search(tool_call)
            elif tool_call.name == "select_voice_line":
                return self._handle_select(tool_call)
            elif tool_call.name == "get_random_greeting":
                return self._handle_random_greeting(tool_call)
            elif tool_call.name == "get_random_farewell":
                return self._handle_random_farewell(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=json.dumps({"error": f"Unknown tool: {tool_call.name}"}),
                )
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({"error": str(e)}),
            )

    def _handle_search(self, tool_call: ToolCall) -> ToolResult:
        """Handle search_voice_lines tool call."""
        args = tool_call.arguments

        desired_text = args.get("desired_text", "")
        category_hint = args.get("category_hint")
        costume_type = args.get("costume_type")

        logger.info(
            f"Searching voice lines: text='{desired_text[:50]}...', "
            f"category={category_hint}, costume={costume_type}"
        )

        # Perform search
        results = self.tts.search_lines(
            desired_text=desired_text,
            category_hint=category_hint,
            costume_type=costume_type,
            n_results=5,
        )

        # Store for potential auto-select
        self._pending_search_results = results

        # Return results
        return ToolResult(
            tool_call_id=tool_call.id,
            content=json.dumps({
                "results": results,
                "count": len(results),
                "suggestion": results[0]["id"] if results else None,
            }),
        )

    def _handle_select(self, tool_call: ToolCall) -> ToolResult:
        """Handle select_voice_line tool call."""
        args = tool_call.arguments
        line_id = args.get("line_id", "")

        logger.info(f"Selecting voice line: {line_id}")

        # Get the line
        line = self.tts.select_line(line_id)

        if line:
            self._selected_line = VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method="search",
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": True,
                    "line_id": line.id,
                    "text": line.text,
                    "message": "Voice line selected and ready to play.",
                }),
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": False,
                    "error": f"Voice line not found: {line_id}",
                }),
            )

    def _handle_random_greeting(self, tool_call: ToolCall) -> ToolResult:
        """Handle get_random_greeting tool call."""
        db = self.tts.get_db()
        line = db.get_random_from_category("greetings")

        if line:
            db.mark_used(line.id)
            self._selected_line = VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method="random",
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": True,
                    "line_id": line.id,
                    "text": line.text,
                }),
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": False,
                    "error": "No greetings available",
                }),
            )

    def _handle_random_farewell(self, tool_call: ToolCall) -> ToolResult:
        """Handle get_random_farewell tool call."""
        db = self.tts.get_db()
        line = db.get_random_from_category("farewells")

        if line:
            db.mark_used(line.id)
            self._selected_line = VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method="random",
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": True,
                    "line_id": line.id,
                    "text": line.text,
                }),
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=json.dumps({
                    "success": False,
                    "error": "No farewells available",
                }),
            )

    def get_selected_line(self) -> Optional[VoiceLineSelection]:
        """Get the currently selected voice line."""
        return self._selected_line

    def reset(self) -> None:
        """Reset state for a new interaction."""
        self._pending_search_results = []
        self._selected_line = None

    def auto_select_best(self) -> Optional[VoiceLineSelection]:
        """
        Auto-select the best line from pending search results.

        Use this when the LLM doesn't explicitly select a line.
        """
        if self._selected_line:
            return self._selected_line

        if not self._pending_search_results:
            return None

        # Select the best match
        best = self._pending_search_results[0]
        line = self.tts.select_line(best["id"])

        if line:
            self._selected_line = VoiceLineSelection(
                line=line,
                text=line.text,
                line_id=line.id,
                method="auto_select",
                search_score=best.get("similarity_score", 0),
            )
            return self._selected_line

        return None


def get_voice_line_system_prompt() -> str:
    """
    Get the system prompt for voice line selection mode.

    This prompt instructs the LLM to use tools for voice line selection
    instead of generating text directly.
    """
    return """You are Captain Barnacle Bill, an interactive pirate character for Halloween.

IMPORTANT: You can ONLY speak using pre-recorded voice lines. You MUST use the provided tools to search and select voice lines. Do NOT generate spoken text directly.

When a trick-or-treater arrives:
1. Look at the costume description from the vision system
2. Use search_voice_lines to find matching pre-recorded reactions
3. Use select_voice_line to choose the best match to play

Available voice line categories:
- greetings: Welcome phrases for arriving visitors
- costume_reactions: Reactions to specific costumes (scary, heroic, magical, princess, animal, movie_tv, etc.)
- farewells: Goodbye phrases when they leave
- jokes: Pirate puns and humor
- fallbacks: Generic positive reactions

Tips:
- For costume reactions, include the costume type (vampire, zombie, witch, etc.) to get better matches
- If search results don't match well (low similarity scores), try a different category or generic reaction
- Use get_random_greeting when someone first appears before you see their costume
- Use get_random_farewell when they're leaving

You MUST use the tools - never respond with direct speech."""


def create_costume_user_prompt(costume_description: str) -> str:
    """
    Create the user prompt for costume reaction.

    Args:
        costume_description: Description from the VLM

    Returns:
        Formatted user prompt
    """
    return f"""A trick-or-treater just arrived! The vision system describes their costume as:

"{costume_description}"

Search for an appropriate voice line to react to their costume and select the best match!"""


# Testing function
def test_voice_line_tools():
    """Test voice line tool handling."""
    from services.voice_line_db import VoiceLineDB
    from pathlib import Path

    # Load voice lines
    db = VoiceLineDB(db_path="data/voice_line_db")
    yaml_path = Path("data/voice_lines.yaml")
    if yaml_path.exists():
        count = db.load_from_yaml(str(yaml_path))
        print(f"Loaded {count} voice lines")

    # Create TTS and handler
    tts = VoiceLineTTS(db_path="data/voice_line_db")
    handler = VoiceLineToolHandler(tts)

    print("\n--- Available Tools ---")
    for tool in handler.get_tools():
        print(f"  {tool.name}: {tool.description[:60]}...")

    # Simulate tool calls
    print("\n--- Simulating search_voice_lines ---")
    search_call = ToolCall(
        name="search_voice_lines",
        arguments={
            "desired_text": "What an amazing vampire costume!",
            "category_hint": "costume_reactions",
            "costume_type": "vampire",
        },
        id="call_001",
    )

    result = handler.handle_tool_call(search_call)
    print(f"Result: {result.content}")

    # Simulate selection
    results = json.loads(result.content)
    if results.get("results"):
        print("\n--- Simulating select_voice_line ---")
        select_call = ToolCall(
            name="select_voice_line",
            arguments={"line_id": results["results"][0]["id"]},
            id="call_002",
        )

        result = handler.handle_tool_call(select_call)
        print(f"Result: {result.content}")

        # Check selected line
        selected = handler.get_selected_line()
        if selected:
            print(f"\nSelected: {selected.line_id}")
            print(f"Text: {selected.text}")

    # Test random greeting
    print("\n--- Testing get_random_greeting ---")
    handler.reset()
    greeting_call = ToolCall(
        name="get_random_greeting",
        arguments={},
        id="call_003",
    )
    result = handler.handle_tool_call(greeting_call)
    print(f"Result: {result.content}")


if __name__ == "__main__":
    test_voice_line_tools()
