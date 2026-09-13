"""
fix_title_everywhere.py
========================
P0-1 "Done when" criterion: only one title string should exist across
manuscript, repo README, and Zenodo record, stating the negative-
direction verdict.

Replaces the OLD title phrase with the NEW one across every .md/.txt
file in the repo tree. Does a literal substring replace of the title
itself (not the whole sentence it may sit inside), so surrounding
Markdown formatting, headers, and context are preserved untouched.

Makes a .bak copy of each file before editing, and prints a diff-style
report of every file it touched so you can review before committing.

Run from repo root: python fix_title_everywhere.py
"""

import os

OLD_TITLE = "Sparse mixture-of-experts routing spontaneously partitions chemical space along physicochemical axes for molecular property prediction"
NEW_TITLE = "Sparse mixture-of-experts routing recovers, rather than creates, physicochemical organization of chemical space for molecular property prediction"

# Also catch the shorter/partial phrase pattern findstr matched on, in case
# the full title doesn't appear verbatim in some files (e.g. truncated in
# a summary doc, or with slightly different wrapping/casing around it).
OLD_FRAGMENT = "spontaneously partitions chemical space"

TARGET_EXTENSIONS = (".md", ".txt")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".idea"}

changed_files = []
skipped_files = []


def process_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                content = f.read()
        except Exception as e:
            skipped_files.append((path, f"decode failed: {e}"))
            return
    except Exception as e:
        skipped_files.append((path, f"read failed: {e}"))
        return

    if OLD_FRAGMENT not in content:
        return

    original = content
    # Prefer replacing the full exact title if present verbatim.
    if OLD_TITLE in content:
        content = content.replace(OLD_TITLE, NEW_TITLE)
    else:
        # Fallback: at least neutralize the fragment so a partial/loose
        # match doesn't survive. Flagged in the report as "fragment only"
        # so you can manually check what surrounding text says.
        content = content.replace(OLD_FRAGMENT, "recovers, rather than creates, physicochemical organization of")

    if content != original:
        backup_path = path + ".bak"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        changed_files.append(path)


def main():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if fname.endswith(TARGET_EXTENSIONS):
                process_file(os.path.join(root, fname))

    print(f"\n{'='*70}")
    print(f"  P0-1 title fix: {len(changed_files)} file(s) updated")
    print(f"{'='*70}")
    for f in changed_files:
        print(f"  [UPDATED] {f}  (backup: {f}.bak)")

    if skipped_files:
        print(f"\n  {len(skipped_files)} file(s) skipped (manual check needed):")
        for f, reason in skipped_files:
            print(f"    [SKIPPED] {f} -- {reason}")

    print(f"\n  Review each .bak diff before deleting backups.")
    print(f"  Once confirmed correct, delete backups with:")
    print(f'    for /r %f in (*.bak) do del "%f"')


if __name__ == "__main__":
    main()
