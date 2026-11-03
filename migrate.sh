#!/usr/bin/env bash
# migrate.sh - copy per-project data from an older quant_agent version folder
# into this one, so a fresh unzip is immediately runnable without re-staging.
#
#     ./migrate.sh /path/to/quant_agent_v11.3
#
# Copies exactly the things the packaging norms strip from shipped zips:
#   keys.local.py            your API keys (never shipped)
#   data/api_cache/*.csv     frozen point-in-time price snapshots
#   data/docs/               your period documents (filings, notes)
#   data/knowledge/          your methodology library
#   eval/raw_prices/         acquired eval price data
#   eval/manifest/           eval manifests built from acquired data
#   eval/golden_dataset/snapshots/   golden-set PIT snapshots (if present)
#   data/outputs/            charts and artifacts (optional history)
#
# Deliberately NOT copied:
#   data/chroma/     rebuilt automatically on first doc_search (index, not data)
#   data/traces/     belong to the old version's runs; keep them there
#   __pycache__ etc. never migrate bytecode across versions
#
# Idempotent and non-destructive: existing files in THIS folder are never
# overwritten (cp -n), so re-running is safe and local changes win.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OLD="${1:-}"

if [[ -z "$OLD" || ! -d "$OLD" ]]; then
    echo "usage: ./migrate.sh /path/to/old/quant_agent_vX.Y" >&2
    echo "       (old folder must exist; run from inside the NEW version folder)" >&2
    exit 1
fi
OLD="$(cd "$OLD" && pwd)"
if [[ "$OLD" == "$HERE" ]]; then
    echo "old and new folders are the same ($HERE) - nothing to do" >&2
    exit 1
fi
if [[ ! -f "$OLD/quant_agent/config.py" ]]; then
    echo "warning: $OLD does not look like a quant_agent folder (no quant_agent/config.py)" >&2
fi

copied=0
skipped=0

copy_file() {  # copy_file <src> <dst>
    if [[ -f "$1" ]]; then
        mkdir -p "$(dirname "$2")"
        if [[ -e "$2" ]]; then
            skipped=$((skipped+1))
        else
            cp -p "$1" "$2" && copied=$((copied+1)) && echo "  + ${2#$HERE/}"
        fi
    fi
}

copy_tree() {  # copy_tree <src_dir> <dst_dir>  (no overwrite, skips placeholders)
    [[ -d "$1" ]] || return 0
    while IFS= read -r -d '' f; do
        rel="${f#$1/}"
        case "$rel" in
            put-*-here.txt|.gitkeep) continue ;;  # placeholders ship with every version
        esac
        copy_file "$f" "$2/$rel"
    done < <(find "$1" -type f -print0)
}

echo "migrating from: $OLD"
echo "            to: $HERE"
echo

copy_file "$OLD/keys.local.py"                 "$HERE/keys.local.py"
copy_tree "$OLD/data/api_cache"                "$HERE/data/api_cache"
copy_tree "$OLD/data/docs"                     "$HERE/data/docs"
copy_tree "$OLD/data/knowledge"                "$HERE/data/knowledge"
copy_tree "$OLD/data/portfolios"               "$HERE/data/portfolios"
copy_tree "$OLD/data/outputs"                  "$HERE/data/outputs"
copy_tree "$OLD/eval/raw_prices"               "$HERE/eval/raw_prices"
copy_tree "$OLD/eval/manifest"                 "$HERE/eval/manifest"
copy_tree "$OLD/eval/golden_dataset/snapshots" "$HERE/eval/golden_dataset/snapshots"

echo
echo "done: $copied file(s) copied, $skipped already present (left untouched)."
echo "note: data/chroma was NOT copied - the index rebuilds itself on first search."
if [[ ! -f "$HERE/keys.local.py" ]]; then
    echo "note: no keys.local.py found in old folder - run.py will create one to fill in."
fi
