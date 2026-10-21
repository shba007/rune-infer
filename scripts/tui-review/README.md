# TUI review frames

Export synthetic ratatui TestBackend frames without running a model. From the
repository root, set `RUNE_TUI_SNAPSHOT_DIR` to one of
`scripts/tui-review/color`, `scripts/tui-review/no-color` or
`scripts/tui-review/ascii`, then run:

```text
cargo test --lib tui::dashboard::tests::export_review_frames -- --exact
```

For color, unset `NO_COLOR` and `RUNE_ASCII`. For monochrome, set `NO_COLOR=1`.
For ASCII, set both `NO_COLOR=1` and `RUNE_ASCII=1`. Export assertions check the
cell colors and glyphs. Generated JSON, PNG and text dumps are ignored.

On Windows, rasterize all exported modes with Python and Pillow:

```text
python scripts/tui-review/render.py
```

The renderer uses Consolas and Segoe UI Symbol from Windows Fonts. PNG font
coverage and cell spacing are approximations; text dumps retain the exact cells.
These frames use synthetic telemetry, not live GPU captures. Busy represents
completed traffic, not a distinct in-flight state. Rendering does not establish
terminal UX quality or live inference performance.
