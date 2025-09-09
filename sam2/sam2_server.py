#!/usr/bin/env python3
import os, time, threading, itertools
import cv2
import argparse
import torch
import numpy as np
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from sam2.build_sam import build_sam2_camera_predictor

# ======== KNOBS ========
INFER_LONG_SIDE = 640
FRAME_SKIP = 1
FILL_HOLES = False
DENSE_STEP = 2
MAX_POINTS = 6000
# =======================

torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="sam2.1", choices=["sam2","sam2.1"])
parser.add_argument("--model_size",    type=str, default="tiny", choices=["tiny","small","base","large"])
parser.add_argument("--camera",        type=int, default=0)
parser.add_argument("--sock_host",     type=str, default="127.0.0.1")
parser.add_argument("--sock_port",     type=int, default=5001)
args = parser.parse_args()

size_map = {
    "tiny":  ("_hiera_tiny.pt",  "_hiera_t.yaml"),
    "small": ("_hiera_small.pt", "_hiera_s.yaml"),
    "base":  ("_hiera_base.pt",  "_hiera_b.yaml"),
    "large": ("_hiera_large.pt", "_hiera_l.yaml"),
}
ckpt_suffix, cfg_suffix = size_map[args.model_size]
model_version = args.model_version
sam2_checkpoint = f"./checkpoints/{model_version}{ckpt_suffix}"
model_cfg       = f"configs/{model_version}/{model_version}{cfg_suffix}"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# ----- Socket server -----
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.get("/health")
def health():
    return jsonify(ok=True, model=model_version, size=args.model_size)

# style/scene broadcasting
current_scene = "calm"
current_style = {}
SCENES = {
    "calm":   {"stroke":"#00e5ff","fill":"rgba(0,229,255,0.18)","lw":3,"glow":6,"blend":"screen"},
    "forest": {"stroke":"#7bd389","fill":"rgba(123,211,137,0.22)","lw":4,"glow":10,"blend":"multiply"},
    "neon":   {"stroke":"#ff006e","fill":"rgba(255,0,110,0.18)","lw":5,"glow":20,"blend":"lighter"},
    "flame":  {"stroke":"#ffd166","fill":"rgba(255,209,102,0.20)","lw":6,"glow":24,"blend":"screen"},
    "aurora": {"stroke":"#b692ff","fill":"rgba(182,146,255,0.20)","lw":4,"glow":16,"blend":"screen"},
}

@app.post("/scene")
def set_scene_api():
    global current_scene
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name")
    if not isinstance(name, str):
        return jsonify(ok=False, error="missing 'name'"), 400
    current_scene = name
    socketio.emit("scene", {"name": name})
    return jsonify(ok=True)

@app.post("/style")
def set_style_api():
    global current_style
    data = request.get_json(force=True, silent=True) or {}
    current_style.update(data)
    socketio.emit("style", current_style)
    return jsonify(ok=True)

@socketio.on("connect")
def on_connect():
    print("[SOCKET] client connected")

def run_socketio():
    socketio.run(app, host=args.sock_host, port=args.sock_port, allow_unsafe_werkzeug=True)

threading.Thread(target=run_socketio, daemon=True).start()
print(f"[SOCKET] ws://{args.sock_host}:{args.sock_port}  events: 'mask','style','scene'")

# ----- Box UI -----
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
bbox = None
enter_pressed = False
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bbox, enter_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; ix, iy = x, y; fx, fy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False; fx, fy = x, y
        bbox = (ix, iy, fx, fy); enter_pressed = True

# ----- Camera -----
try:
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
except:
    cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.setNumThreads(1)

cv2.namedWindow("SAM2 Preview")
cv2.setMouseCallback("SAM2 Preview", draw_rectangle)

if_init = False
last_vis = None
frm_idx = 0

# emit default scene/style (harmless if client ignores)
socketio.emit("scene", {"name": current_scene})
socketio.emit("style", SCENES[current_scene])

with torch.inference_mode():
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame"); break

        # NOTE: no mirroring here; frontend can flip if needed
        h, w = frame.shape[:2]

        # downscale for inference
        long_side = max(h, w)
        scale = INFER_LONG_SIDE / float(long_side)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            infer_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sx, sy = w / float(new_w), h / float(new_h)
        else:
            infer_frame = frame; sx = sy = 1.0

        # wait for box
        if not enter_pressed:
            vis = frame.copy()
            if drawing and ix >= 0 and iy >= 0:
                cv2.rectangle(vis, (ix, iy), (fx, fy), (255, 0, 0), 2)
            cv2.putText(vis, "Draw a box (q=quit, r=reset)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("SAM2 Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            if key == ord('r'):
                enter_pressed = False; last_vis = None; continue
            continue

        run_now = (frm_idx % (FRAME_SKIP + 1) == 0)

        if not if_init:
            if_init = True
            predictor.load_first_frame(infer_frame)
            x1, y1, x2, y2 = bbox
            sbbox = np.array([[x1 / sx, y1 / sy], [x2 / sx, y2 / sy]], dtype=np.float32)
            _ , out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=(1,), bbox=sbbox
            )

        elif run_now:
            out_obj_ids, out_mask_logits = predictor.track(infer_frame)

            out_mask = (out_mask_logits[0] > 0.0).permute(1,2,0).contiguous()[...,0].to(torch.uint8)
            mask_np = out_mask.to("cpu", non_blocking=True).numpy()

            if scale < 1.0:
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            if FILL_HOLES:
                if mask_np.max() == 1:
                    mask_np = (mask_np * 255).astype(np.uint8)
                kernel = np.ones((5,5), np.uint8)
                mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=1)
                mask_np = (mask_np > 0).astype(np.uint8)

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            polys = []
            for cnt in contours:
                pts = cnt.squeeze(1)
                if pts.ndim != 2 or pts.shape[0] < 3: continue
                if DENSE_STEP > 1: pts = pts[::DENSE_STEP]
                if len(pts) > MAX_POINTS:
                    stride = max(1, len(pts)//MAX_POINTS)
                    pts = pts[::stride]
                # normalize to [0..1]
                norm = [[float(x)/w, float(y)/h] for (x,y) in pts]
                if len(norm) >= 3: polys.append(norm)

            socketio.emit("mask", {"t": time.time(), "polys": polys})
            # local preview
            vis = frame.copy()
            if np.any(mask_np): cv2.drawContours(vis, contours, -1, (0,255,0), 2)
            last_vis = vis

        show = last_vis if last_vis is not None else frame
        cv2.imshow("SAM2 Preview", show)
        frm_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key == ord('r'):
            if_init = False; enter_pressed = False; last_vis = None

cap.release()
cv2.destroyAllWindows()
