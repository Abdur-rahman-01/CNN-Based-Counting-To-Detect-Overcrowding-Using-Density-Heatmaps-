import cv2
import numpy as np
import torch
from torchvision import transforms

WEIGHTS_PATH = "weights/csrnet_v3_best.pth"
FONT         = cv2.FONT_HERSHEY_SIMPLEX

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


def load_model(weights_path=WEIGHTS_PATH):
    from model import CSRNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CSRNet()
    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model, device


def predict_frame(model, device, frame_rgb):
    tensor = transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        density = model(tensor).squeeze().cpu().numpy()
    density = np.maximum(density, 0)
    count   = int(density.sum())
    return count, density


def get_alert(count):
    if   count < 30:  return "NORMAL",      (34,  197, 94 )
    elif count < 70:  return "MODERATE",    (234, 179, 8  )
    elif count < 120: return "CROWDED",     (249, 115, 22 )
    else:             return "OVERCROWDED", (239, 68,  68 )


def normalise(density, count):
    if density.max() < 1e-8:
        return np.zeros_like(density, dtype=np.uint8)
    if count < 15:
        out = density / density.max() * 80
    elif count < 50:
        out = density / density.max() * 160
    else:
        p2  = np.percentile(density, 2)
        p98 = np.percentile(density, 98)
        out = np.clip((density - p2) / (p98 - p2 + 1e-8), 0, 1) * 255
    return out.astype(np.uint8)


def draw_hud(frame_bgr, count, density, alpha=0.45):
    h, w = frame_bgr.shape[:2]

    # resize density map → frame size
    hmap       = cv2.resize(density, (w, h))
    hmap       = normalise(hmap, count)
    hmap_color = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    blended    = cv2.addWeighted(frame_bgr, 1 - alpha,
                                 hmap_color, alpha, 0)

    # black HUD bar at top
    bar_h = 56
    cv2.rectangle(blended, (0, 0), (w, bar_h), (10, 10, 10), -1)

    alert, rgb = get_alert(count)
    bgr        = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    # coloured alert pill
    cv2.rectangle(blended, (8, 8), (178, bar_h - 8), bgr, -1)
    cv2.putText(blended, alert,
                (16, bar_h - 16), FONT, 0.56,
                (0, 0, 0), 2, cv2.LINE_AA)

    # count text
    cv2.putText(blended, f"Count: {count}",
                (192, bar_h - 14), FONT, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA)

    # capacity
    cap_pct = min(int(count / 200 * 100), 100)
    cv2.putText(blended, f"Capacity: {cap_pct}%",
                (390, bar_h - 14), FONT, 0.58,
                (180, 180, 180), 1, cv2.LINE_AA)

    return blended


def process_video(model, device,
                  input_path, output_path,
                  frame_skip=1, alpha=0.45,
                  progress_callback=None):
    """
    Reads input_path frame by frame, overlays heatmap,
    writes to output_path. Returns stats dict.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    idx         = 0
    last_frame  = None
    total_count = 0
    peak_count  = 0
    inferred    = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if idx % frame_skip == 0:
            rgb            = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            count, density = predict_frame(model, device, rgb)
            last_frame     = draw_hud(bgr, count, density, alpha)
            total_count   += count
            peak_count     = max(peak_count, count)
            inferred      += 1
        else:
            if last_frame is None:
                last_frame = bgr

        writer.write(last_frame)
        idx += 1

        if progress_callback and total > 0:
            progress_callback(idx / total)

    cap.release()
    writer.release()

    return {
        "total_frames" : idx,
        "inferred"     : inferred,
        "avg_count"    : total_count // max(inferred, 1),
        "peak_count"   : peak_count,
    }