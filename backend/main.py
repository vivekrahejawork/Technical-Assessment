# backend/main.py
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from dotenv import load_dotenv
import logging, os, hashlib, urllib.request
import cv2
import ffmpeg
import numpy as np
import mediapipe as mp  # pip install mediapipe
import yt_dlp  # pip install yt-dlp

app = Flask(__name__)
CORS(app)
load_dotenv()

# -------------------- logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# -------------------- paths ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# bump when changing algo/codec to avoid stale cache reuse
CACHE_VERSION = "h264_mediapipe_pipe_pose_v4"  # v4: less halo (threshold↑, dilation↓, feather↓, alpha sharpen, faster EMA)

# -------------------- MediaPipe models -----------
mp_selfie = mp.solutions.selfie_segmentation
SEG_MODEL = mp_selfie.SelfieSegmentation(model_selection=1)  # 0: landscape, 1: general

mp_pose = mp.solutions.pose
POSE_MODEL = mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False)

# -------------------- temporal smoothing state ---
_prev_alpha = None          # HxWx1 float32 in [0,1]
_prev_size  = None          # (h, w)


# -------------------- helpers --------------------
def download_video(url: str, output_path: str) -> bool:
    """
    Download video from URL (supports YouTube and direct links).
    Returns True if successful, False otherwise.
    """
    try:
        if 'youtube.com' in url or 'youtu.be' in url:
            # Use yt-dlp for YouTube videos
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[ext=mp4]',  # Prefer mp4 format
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
        else:
            # Use urllib for direct video links
            urllib.request.urlretrieve(url, output_path)
            return True
    except Exception as e:
        logger.error(f"Failed to download video from {url}: {e}")
        return False


# -------------------- routes ---------------------
@app.route("/hello-world", methods=["GET"])
def hello_world():
    return jsonify({"Hello": "World"}), 200


@app.route("/processed-video", methods=["GET"])
def processed_video():
    """
    GET /processed-video?src=<remote_mp4_url>
    - Downloads remote MP4 (cached by URL hash).
    - Builds color-foreground / grayscale-background composite per frame.
    - Streams raw frames directly to ffmpeg (no AVI) -> H.264 MP4.
    """
    src = request.args.get("src", "").strip()
    if not src:
        return jsonify({"error": "Missing query param 'src' with a video URL"}), 400

    key = hashlib.md5(src.encode("utf-8")).hexdigest()[:16]
    raw_path = os.path.join(CACHE_DIR, f"{key}_raw.mp4")
    out_path = os.path.join(CACHE_DIR, f"{key}_{CACHE_VERSION}.mp4")

    # 1) Download source if not cached
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        logger.info(f"[download] fetching {src} -> {raw_path}")
        if not download_video(src, raw_path):
            return jsonify({"error": "Failed to download video"}), 502

    # 2) Process + encode if not cached for this version
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        logger.info(f"[process] starting (pipe->ffmpeg) -> {out_path}")
        try:
            process_video_to_h264_pipe(raw_path, out_path)
            logger.info("[process] done")
        except Exception as e:
            logger.exception("Processing failed")
            return jsonify({"error": f"Processing failed: {e}"}), 500

    # 3) Serve processed MP4 with explicit headers
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return jsonify({"error": "Processed file is empty"}), 500

    size = os.path.getsize(out_path)
    resp = send_file(out_path, mimetype="video/mp4", as_attachment=False, conditional=True)
    resp.headers["Content-Type"] = "video/mp4"
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(size)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.route("/preview", methods=["GET"])
def preview_mjpeg():
    """
    Streams processed frames as MJPEG (codec-free) for quick visual verification.
    GET /preview?src=<remote_mp4_url>
    """
    src = request.args.get("src", "").strip()
    if not src:
        return jsonify({"error": "Missing query param 'src'"}), 400

    key = hashlib.md5(src.encode("utf-8")).hexdigest()[:16]
    raw_path = os.path.join(CACHE_DIR, f"{key}_raw.mp4")
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        logger.info(f"[preview] downloading {src} -> {raw_path}")
        if not download_video(src, raw_path):
            return jsonify({"error": "Failed to download video for preview"}), 502

    def gen():
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            logger.error(f"Could not open {raw_path} for preview")
            return

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            composite = apply_person_mask_composite(frame)

            ok, jpg = cv2.imencode(".jpg", composite)
            if not ok:
                break

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
            idx += 1
            if idx % 60 == 0:
                logger.info(f"[preview] streamed {idx} frames")

        cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# -------------------- mask helpers ----------------
def _feather_alpha(binary_mask_0_255: np.ndarray, feather_px: int = 3) -> np.ndarray:
    """
    Build a crisp alpha with a tiny feather band using distance transforms.
    Returns float32 alpha in [0,1].
    """
    m = (binary_mask_0_255 > 127).astype(np.uint8)
    inv = 1 - m
    dist_in  = cv2.distanceTransform(m,   distanceType=cv2.DIST_L2, maskSize=3)
    dist_out = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    signed = dist_in - dist_out  # positive inside, negative outside
    a = (signed + feather_px) / (2.0 * feather_px)  # map [-f,+f] -> [0,1]
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return a


def _pose_arm_mask(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Draw thick strokes for upper/lower arms from MediaPipe Pose.
    Returns uint8 mask in {0,255}. Confidence-gated.
    """
    res = POSE_MODEL.process(rgb)
    arm = np.zeros((h, w), np.uint8)
    if not getattr(res, "pose_landmarks", None):
        return arm

    lm = res.pose_landmarks.landmark

    def ok(idx):
        vis = getattr(lm[idx], "visibility", 1.0)
        prs = getattr(lm[idx], "presence",   1.0)
        return (vis >= 0.6) and (prs >= 0.5)

    def xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    pairs = [(11, 13), (13, 15), (12, 14), (14, 16)]
    R = max(10, int(0.014 * min(h, w)))  # ~1.4% of min dimension

    for a, b in pairs:
        if not (ok(a) and ok(b)):
            continue
        x1, y1 = xy(a)
        x2, y2 = xy(b)
        cv2.line(arm, (x1, y1), (x2, y2), 255, thickness=2 * R)
        cv2.circle(arm, (x1, y1), R, 255, -1)
        cv2.circle(arm, (x2, y2), R, 255, -1)

    return arm


# -------------------- segmentation & compositing --
def apply_person_mask_composite(frame_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe SelfieSegmentation -> person alpha in [0,1].
    Robust to NaNs/Infs; returns BGR with person in color, background in grayscale.
    """
    h, w = frame_bgr.shape[:2]

    # 1) Run segmentation (expects RGB)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = SEG_MODEL.process(rgb)

    seg = getattr(res, "segmentation_mask", None)
    if seg is None:
        # Safe fallback: whole frame grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2) Initial binary person mask (tweak threshold as needed)
    # Get frame dimensions
    height, width = rgb.shape[:2]
    
    # Get person mask with slightly expanded boundaries
    person_bin = (seg > 0.5).astype(np.uint8) * 255  # slightly more inclusive threshold
    
    # Run pose detection for core body parts only
    pose_mask = _pose_arm_mask(rgb, height, width)
    person_bin = cv2.bitwise_or(person_bin, pose_mask)
    
    # 3) Conservative morphology to keep only core areas
    # Use minimal kernels to avoid edge expansion
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # minimal opening
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # small closing
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # aggressive erosion
    
    # Remove small noise first
    person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_OPEN, open_k, iterations=2)
    
    # Fill small holes in core body
    person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_CLOSE, close_k, iterations=1)
    
    # Moderately erode to clean edges while preserving boundaries
    person_bin = cv2.erode(person_bin, erode_k, iterations=1)

    # 4) Feather edges: distance transform -> normalized soft edge
    inv = cv2.bitwise_not(person_bin)
    dist_in  = cv2.distanceTransform(person_bin, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(inv,         cv2.DIST_L2, 3)

    # Create near-binary mask with minimal feathering
    feather = 2.0  # very minimal feathering
    alpha = dist_in / (dist_in + dist_out + 1e-6)
    alpha = np.power(alpha, 2.5)  # very sharp transition curve
    alpha = (alpha * 255.0).astype(np.uint8)

    # 5) No blur - keep edges sharp
    # alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # commented out for sharper edges

    # --- 6) sanitize and clip ---
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # --- 7) moderate shrinking to clean edges while preserving boundaries ---
    shrink_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_shrunk_u8 = cv2.erode((alpha * 255).astype(np.uint8), shrink_k, iterations=2)
    alpha = alpha_shrunk_u8.astype(np.float32) / 255.0

    # --- 8) no smoothing to maintain hard edges ---
    # Skip bilateral filter to keep edges sharp

    # --- 9) composite ---
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha3 = alpha[..., None]
    out = (alpha3 * frame_bgr + (1.0 - alpha3) * gray_bgr).astype(np.uint8)

    # --- 10) Binary separation - no halo, no transition zone ---
    # Create pure binary mask with slightly expanded boundaries
    binary_threshold = 0.95  # high threshold but slightly more inclusive
    
    # Convert everything below threshold to pure grayscale
    bg_mask = (alpha < binary_threshold)
    if np.any(bg_mask):
        # Pure neutral grayscale conversion
        pure_gray = cv2.cvtColor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out[bg_mask] = pure_gray[bg_mask]
    
    # Keep only the most certain person pixels in color
    person_mask = (alpha >= binary_threshold)
    if np.any(person_mask):
        out[person_mask] = frame_bgr[person_mask]
    
    # No transition zone - binary cutoff only

    return out




# -------------------- processing (ffmpeg pipe) ----
def process_video_to_h264_pipe(input_mp4: str, output_mp4: str):
    """
    Read frames with OpenCV, apply composite, and pipe raw BGR frames
    directly to ffmpeg to produce H.264 MP4. No intermediate AVI.
    """
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 30.0
        logger.info("FPS reported as 0; defaulting to 30.0")
    if width <= 0 or height <= 0:
        ok, fr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("No frames to infer size")
        height, width = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info(f"Inferred size=({width}x{height}) from first frame")

    # Start ffmpeg process that reads rawvideo from stdin and encodes H.264
    # Also copy the audio stream from the input video
    video_stream = ffmpeg.input(
        'pipe:',
        format='rawvideo',
        pix_fmt='bgr24',
        s=f'{width}x{height}',
        r=fps
    )
    
    # Add input video for audio
    audio_stream = ffmpeg.input(input_mp4)
    
    proc = (
        ffmpeg
        .output(
            video_stream,
            audio_stream.audio,  # Only take audio from second input
            output_mp4,
            vcodec='libx264',
            pix_fmt='yuv420p',
            movflags='+faststart',
            preset='veryfast',
            crf=23,
            r=fps,
            acodec='copy'  # Copy audio codec as is
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            composite = apply_person_mask_composite(frame)

            if not composite.flags['C_CONTIGUOUS']:
                composite = np.ascontiguousarray(composite)

            proc.stdin.write(composite.tobytes())
            frames += 1
            if frames % 60 == 0:
                logger.info(f"[process] {frames} frames")
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode('utf-8', errors='ignore') if proc.stderr else ""
        retcode = proc.wait()
        if retcode != 0:
            logger.error(err)
            raise RuntimeError(f"ffmpeg returned non-zero exit code: {retcode}")
        size = os.path.getsize(output_mp4) if os.path.exists(output_mp4) else 0
        logger.info(f"[ffmpeg] wrote H.264 MP4: {output_mp4} ({size} bytes)")


# -------------------- entrypoint -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
