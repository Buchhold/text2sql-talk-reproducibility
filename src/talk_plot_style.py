"""Shared visual tokens for the talk plots."""

TEXT = "#20242C"
MUTED = "#5B5B66"
GRID = "#E4E4EA"

# Model identity, reused across every comparison plot.
MODEL_COLORS = {
    "qwen": "#1F8FE0",      # CID blue: primary open-weight SFT story
    "gemma": "#C14C8B",     # magenta: second open-weight family
    "gemini": "#D6A32D",    # gold — clearly distinct from Qwen blue
    "glm": "#4C9A7A",       # teal-green
    "grok": "#C15F3C",      # CID orange
    "vanilla": "#B9B9C6",   # neutral baseline
}


def light_tint(hex_color: str, white_share: float = 0.58) -> tuple[float, float, float]:
    """Blend a hex colour with white while preserving its model identity."""
    rgb = tuple(int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return tuple((channel + (255 - channel) * white_share) / 255 for channel in rgb)