import re
import json
import click
from glob import glob
from copy import deepcopy

from ocrd_calamari.util import working_directory


@click.command
@click.argument("checkpoint_dir")
def fix_calamari1_model(checkpoint_dir):
    """
    Fix old Calamari 1 models.

    This currently means fixing regexen in "replacements" to have their global flags
    in front of the rest of the regex.
    """
    with working_directory(checkpoint_dir):
        for fn in glob("*.json"):
            with open(fn, "r") as fp:
                j = json.load(fp)
                old_j = deepcopy(j)

            for v in j["model"].values():
                if type(v) != dict:
                    continue
                for child in v.get("children", []):
                    for replacement in child.get("replacements", []):
                        # Move global flags in front
                        replacement["old"] = re.sub(
                            r"^(.*)\(\?u\)$", r"(?u)\1", replacement["old"]
                        )

            if j == old_j:
                print(f"{fn} unchanged.")
            else:
                with open(fn, "w") as fp:
                    json.dump(j, fp, indent=2)
                print(f"{fn} fixed.")
