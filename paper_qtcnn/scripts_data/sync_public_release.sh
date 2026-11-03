#!/usr/bin/env bash
# Rebuild the public release (github.com/ChiShengChen/qtcnn-trained-frozen-absent) from the
# private repository's HEAD.
#
#   scripts_data/sync_public_release.sh <checkout of the public repo>
#
# Exports the committed tree minus the Kaggle data subset (not redistributable), the
# literature folder, the private root README (the public repo has its own), the cover letter
# (it names suggested reviewers) and a stray pip
# log, then adds the gitignored daily return series and the built PDF, which the manuscript
# says are released. The public README.md and LICENSE in the destination are kept.
set -euo pipefail
DST=$(cd "${1:?usage: sync_public_release.sh <public checkout>}" && pwd)
SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
[ -d "$DST/.git" ] || { echo "not a git checkout: $DST"; exit 1; }
find "$DST" -mindepth 1 -maxdepth 1 ! -name .git ! -name README.md ! -name LICENSE -exec rm -rf {} +
cd "$SRC"
git archive HEAD -- . ':(exclude)data_subset_stride11' ':(exclude)paper_qtcnn/literature' \
  ':(exclude)README.md' ':(exclude)=1.3.6' ':(exclude)paper_qtcnn/manuscript/cover_letter.txt' \
  | tar -x -C "$DST"
sed -i '/^results\/\*\*\/returns\/$/d' "$DST/paper_qtcnn/.gitignore"
for d in paper_qtcnn/results/*/returns; do
  mkdir -p "$DST/$d" && cp -r "$d"/. "$DST/$d"/
done
cp paper_qtcnn/manuscript/main.pdf "$DST/paper_qtcnn/manuscript/main.pdf"
mkdir -p "$DST/docs" && git show HEAD:README.md > "$DST/docs/legacy_pipelines_README.md"
cd "$DST" && git add -A && git status --short | awk '{print $1}' | sort | uniq -c
