"""Rasterize exported ratatui TestBackend cells; requires Pillow, no terminal capture."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
base = Path(__file__).parent
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
symbol_font = ImageFont.truetype("C:/Windows/Fonts/seguisym.ttf", 16)
colors = {"Reset": "#dedede", "White": "#dedede", "Cyan": "#52d8e8", "DarkGray": "#777777", "Black": "#111111", "Red": "#ef5555", "Gray": "#bbbbbb", "Yellow": "#eeeeee", "Green": "#eeeeee"}
for source in base.glob("*/*.json"):
    frame = json.loads(source.read_text(encoding="utf-8"))
    width, height = frame["width"], frame["height"]
    image = Image.new("RGB", (width*10+24, height*22+24), "#111111")
    draw = ImageDraw.Draw(image)
    lines = []
    for y in range(height):
        line = ""
        for x in range(width):
            cell = frame["cells"][y*width+x]
            text = cell["text"]
            line += text
            position = (12+x*10,12+y*22)
            bg = cell["bg"]
            if bg != "Reset":
                draw.rectangle([*position,position[0]+9,position[1]+21], fill=colors.get(bg,"#111111"))
            draw.text(position, text, font=font if text.isascii() else symbol_font, fill=colors.get(cell["fg"],"#dedede"))
        lines.append(line.rstrip())
    image.save(source.with_suffix(".png"))
    source.with_suffix(".txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
