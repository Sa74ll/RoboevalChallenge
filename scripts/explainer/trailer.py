"""The README's explainer trailer (one GIF cut from the seven explainer videos) and the video gallery's poster tiles.

Run after compose.py has written assets/videos/<Task>.mp4 and assets/controllers/<Task>_fit.png:

  python scripts/explainer/trailer.py --ffmpeg /path/to/ffmpeg
Writes assets/explainers.gif and assets/controllers/<Task>_poster.jpg (plus docs_poster.jpg). Needs numpy, matplotlib
and Pillow.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose import C_FIT, FG, FPS, GOAL, MUTED, MH, MW, MX, MY, H, W  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VID, CTL = REPO / "assets" / "videos", REPO / "assets" / "controllers"
# trailer order, and the control steps of each video's episode (the same episodes as the report's Fig. 2)
STEPS = {"PackBox": 149, "RotateValve": 83, "StackTwoBlocks": 122, "CubeHandover": 81, "LiftTray": 67,
         "PickSingleBookFromTable": 169, "StackSingleBookShelf": 139}
SHORT = {"PickSingleBookFromTable": "PickBook", "StackSingleBookShelf": "ShelfBook"}
ACT_STEP_FRAMES = 3  # compose.py: video frames per control step in the Act chapter
RESULT_FRAMES = 4 * FPS  # compose.py: the result card closes every video
# poster frame: this fraction of the way through the episode
POSTER_AT = {"PackBox": 0.45, "RotateValve": 0.8, "StackTwoBlocks": 0.6, "CubeHandover": 0.75, "LiftTray": 0.45,
             "PickSingleBookFromTable": 0.3, "StackSingleBookShelf": 0.3}
PW, PH = 640, 434  # poster size


def font(size, bold=False):
    return ImageFont.truetype(findfont(FontProperties(family="Noto Sans", weight="bold" if bold else "normal")), size)


def n_frames(mp4, ffmpeg):
    """frame count of a video (decodes it once)"""
    r = subprocess.run([ffmpeg, "-i", str(mp4), "-map", "0:v", "-f", "null", "-"], capture_output=True, text=True)
    return int(r.stderr.rsplit("frame=", 1)[1].split()[0])


def act_frames(task, ffmpeg, every):
    """every `every`-th control step of the video's Act chapter, as full-size RGB frames"""
    mp4 = VID / f"{task}.mp4"
    n = STEPS[task]
    total = n_frames(mp4, ffmpeg)
    a = total - RESULT_FRAMES - ACT_STEP_FRAMES * (n + 1)  # first frame of the Act chapter
    keep = {a + ACT_STEP_FRAMES * t for t in range(0, n + 1, every)} | {a + ACT_STEP_FRAMES * n}
    p = subprocess.Popen([ffmpeg, "-loglevel", "error", "-ss", f"{(a - 0.5) / FPS:.4f}", "-i", str(mp4),
                          "-frames:v", str(ACT_STEP_FRAMES * (n + 1)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    out = []
    for i in range(a, a + ACT_STEP_FRAMES * (n + 1)):
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            break
        if i in keep:
            out.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
    p.wait()
    assert len(out) == len(keep), f"{task}: expected {len(keep)} Act frames, got {len(out)}"
    return out


def gif(frames, path, ffmpeg, fps, width, colors, dither):
    vf = (f"scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle")
    p = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(fps), "-i", "-", "-vf", vf, "-loop", "0", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def poster(view, title, sub, badge, path, icon="play"):
    """a gallery tile: a camera view, a play button (or an arrow), the title and a badge (the video's length)"""
    img = Image.fromarray(view).resize((PW, PH), Image.LANCZOS).convert("RGBA")
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shade)
    for y in range(PH - 220, PH):  # dark gradient behind the text
        d.line([(0, y), (PW, y)], fill=(13, 17, 23, int(235 * min(1.0, (y - PH + 220) / 150))))
    img = Image.alpha_composite(img, shade)
    d = ImageDraw.Draw(img)
    cx, cy, r = PW // 2, PH // 2 - 40, 50
    light = (230, 237, 243, 255)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 17, 23, 200), outline=light, width=4)
    if icon == "play":
        d.polygon([(cx - 16, cy - 26), (cx - 16, cy + 26), (cx + 28, cy)], fill=light)
    else:  # an arrow: a link to a page
        d.line([(cx - 24, cy), (cx + 20, cy)], fill=light, width=8)
        d.polygon([(cx + 4, cy - 20), (cx + 4, cy + 20), (cx + 30, cy)], fill=light)
    d.text((26, PH - 64), title, font=font(38, True), fill=FG, anchor="ls")
    size = 21
    while d.textlength(sub, font=font(size)) > PW - 52 and size > 14:
        size -= 1
    d.text((26, PH - 24), sub, font=font(size), fill=MUTED, anchor="ls")
    f = font(20, True)
    w = d.textlength(badge, font=f)
    d.rounded_rectangle([PW - 26 - w - 24, 22, PW - 26, 58], 10, fill=(13, 17, 23, 215))
    d.text((PW - 26 - w - 12, 40), badge, font=f, fill=C_FIT, anchor="lm")
    img.convert("RGB").save(path, quality=86, optimize=True, progressive=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--out", default=str(REPO / "assets" / "explainers.gif"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--every", type=int, default=6, help="control steps per trailer frame (20 steps = 1 s)")
    ap.add_argument("--fit-s", type=float, default=0.9, help="seconds each perception slide stays up")
    ap.add_argument("--width", type=int, default=880)
    ap.add_argument("--colors", type=int, default=160)
    ap.add_argument("--dither", default="none")
    a = ap.parse_args()
    frames = []
    for task in STEPS:
        fit = np.asarray(Image.open(CTL / f"{task}_fit.png").convert("RGB"))
        acts = act_frames(task, a.ffmpeg, a.every)
        frames += [fit] * round(a.fit_s * a.fps) + acts + [acts[-1]] * round(0.5 * a.fps)
        secs = n_frames(VID / f"{task}.mp4", a.ffmpeg) / FPS
        name = SHORT.get(task, task)
        # the camera view without its time label (the top 44 px)
        view = acts[round(POSTER_AT[task] * (len(acts) - 1))][MY + 44:MY + MH, MX:MX + MW]
        poster(view, name, GOAL[task], f"{int(secs // 60)}:{round(secs % 60):02d}", CTL / f"{task}_poster.jpg")
        print(f"{task}: {len(acts)} Act frames, video {secs:.1f} s", flush=True)
    cloud = np.asarray(Image.open(CTL / "common_points.png").convert("RGB"))[118:630, 100:860]  # the 3-D point view
    poster(cloud, "In depth", "Perception, plan and phases of each controller", "docs", CTL / "docs_poster.jpg",
           icon="arrow")
    gif(frames, a.out, a.ffmpeg, a.fps, a.width, a.colors, a.dither)
    print(f"{a.out}: {len(frames)} frames ({len(frames) / a.fps:.1f} s), {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
