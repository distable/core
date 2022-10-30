# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations

import csv
import os
import os.path
import typing
import tempfile
import shutil
from pathlib import Path

from src_core import paths
from src_core.classes.Plugin import Plugin

if typing.TYPE_CHECKING:
    # Only import this when code is being type-checked, it doesn't have any effect at runtime
    from src_plugins.sd1111_plugin.SDJob import SDJob

class PromptStylePlugin(Plugin):
    def __init__(self, dirpath: Path, id: str = None):
        super().__init__(dirpath, id)
        self.styles = None

    def title(self):
        return "Prompt Styles"

    def load(self):
        self.styles = StyleDatabase(paths.root / 'styles')

    def prompt2prompt(self, txt):
        return txt


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def merge_prompts(style_prompt: str, prompt: str) -> str:
    if "{prompt}" in style_prompt:
        res = style_prompt.replace("{prompt}", prompt)
    else:
        parts = filter(None, (prompt.strip(), style_prompt.strip()))
        res = ", ".join(parts)

    return res


def apply_style(prompt, styles):
    for style in styles:
        prompt = merge_prompts(style, prompt)

    return prompt


class StyleDatabase:
    def __init__(self, path: str):
        self.no_style = PromptStyle("None", "", "")
        self.styles = {"None": self.no_style}

        if not os.path.exists(path):
            return

        # Support CSV  [name|prompt, text]  columns
        # ----------------------------------------
        with open(path, "r", encoding="utf-8-sig", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                prompt = row["prompt"] if "prompt" in row else row["text"]
                promptneg = row.get("promptneg", "")
                self.styles[row["name"]] = PromptStyle(row["name"], prompt, promptneg)

    def get_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).prompt for x in styles]

    def get_negative_style_prompts(self, styles):
        return [self.styles.get(x, self.no_style).negative_prompt for x in styles]

    def apply_style(self, prompt, styles):
        return apply_style(prompt, [self.styles.get(x, self.no_style).prompt for x in styles])

    def apply_neg_styles(self, prompt, styles):
        return apply_style(prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles])

    def apply(self, p: SDJob) -> None:
        if isinstance(p.prompt, list):
            p.prompt = [self.apply_style(prompt, p.styles) for prompt in p.prompt]
        else:
            p.prompt = self.apply_style(p.prompt, p.styles)

        if isinstance(p.promptneg, list):
            p.promptneg = [self.apply_neg_styles(prompt, p.styles) for prompt in p.promptneg]
        else:
            p.promptneg = self.apply_neg_styles(p.promptneg, p.styles)

    def save_styles(self, path: str) -> None:
        # Write to temporary file first, so we don't nuke the file if something goes wrong
        fd, temp_path = tempfile.mkstemp(".csv")
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline='') as file:
            # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
            # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
            writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
            writer.writeheader()
            writer.writerows(style._asdict() for k,     style in self.styles.items())

        # Always keep a backup file around
        if os.path.exists(path):
            shutil.move(path, path + ".bak")
        shutil.move(temp_path, path)
