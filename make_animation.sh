#!/usr/bin/env bash
set -euo pipefail

# 用法：./make_animation.sh [帧率]，默认 10 帧/秒。
# 读取项目 out/knn-*.png，生成并覆盖根目录的 knn.mp4 和 knn.gif。
fps="${1:-10}"
if (( $# > 1 )) || [[ ! "$fps" =~ ^[1-9][0-9]*$ ]]; then
    printf '用法：%s [正整数帧率，默认 10]\n' "$0" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf '错误：请先安装 ffmpeg。\n' >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

shopt -s nullglob
frames=(out/knn-*.png)
if (( ${#frames[@]} == 0 )); then
    printf '错误：out/ 中没有 knn-*.png 图片。\n' >&2
    exit 1
fi

printf '将 %s 张图片以 %s 帧/秒合成 MP4 和 GIF。\n' "${#frames[@]}" "$fps"

# 按编号排序，避免 knn-1000.png 排在 knn-200.png 前面。
printf '%s\0' "${frames[@]}" | sort -zV | xargs -0 cat | \
    ffmpeg -hide_banner -loglevel warning -y \
        -f image2pipe -framerate "$fps" -c:v png -i - \
        -c:v libx264 -pix_fmt yuv420p -movflags +faststart knn.mp4

ffmpeg -hide_banner -loglevel warning -y -i knn.mp4 \
    -vf "fps=${fps},split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    -loop 0 knn.gif

printf '已生成：\n%s/knn.mp4\n%s/knn.gif\n' "$script_dir" "$script_dir"
