import numpy as np
import gradio as gr




def _to_uint8_lut(arr_float: np.ndarray) -> np.ndarray:
   
    arr = np.rint(arr_float).astype(np.float64, copy=False)  # safe rounding
    arr = np.clip(arr, 0, 255)
    arr_u8 = arr.astype(np.uint8)
    arr_u8 = np.asarray(arr_u8).reshape(-1)
    if arr_u8.shape[0] != 256:
        raise ValueError(f"LUT must have length 256, got {arr_u8.shape[0]}")
    return arr_u8


def _ensure_gray_uint8(image: np.ndarray | None) -> np.ndarray | None:

    if image is None:
        return None

    img = np.asarray(image)

 
    if np.issubdtype(img.dtype, np.floating):
      
        m = float(np.nanmax(img)) if img.size else 0.0
        if m <= 1.0:
            img = np.clip(img * 255.0, 0.0, 255.0)
        else:
            img = np.clip(img, 0.0, 255.0)
        img = np.rint(img).astype(np.uint8)

    
    if img.ndim == 3:
       
        if img.shape[2] >= 3:
            r = img[..., 0].astype(np.float64)
            g = img[..., 1].astype(np.float64)
            b = img[..., 2].astype(np.float64)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            img = np.rint(np.clip(gray, 0, 255)).astype(np.uint8)
        else:
            img = np.squeeze(img)
    elif img.ndim != 2:
        img = np.squeeze(img)
        if img.ndim != 2:
            raise ValueError(f"Expected grayscale image shape (H,W); got {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float64), 0, 255)
        img = np.rint(img).astype(np.uint8)

    return img


#----------------------------------------------------|secondary functions^| actual stuff you care about V|--------------------------------



def apply_intensity_transform(image: np.ndarray, int_transform) -> np.ndarray:
    img = _ensure_gray_uint8(image)
    if img is None:
        return None

    lut = np.asarray(int_transform).reshape(-1)

    if lut.dtype != np.uint8:
        lut = np.rint(lut.astype(np.float64))
        lut = np.clip(lut, 0, 255).astype(np.uint8)

    max_val = int(img.max()) if img.size else 0
    if lut.size <= max_val:
        raise ValueError("int_transform LUT is too short for the image's max intensity")

    return lut[img]




def get_log_transform(max_r):
    max_r = float(max_r)

    r = np.arange(256, dtype=np.float64)

    if max_r <= 0:
        s = np.zeros_like(r)
    else:
        c = 255.0 / np.log(1.0 + max_r)
        s = c * np.log(1.0 + r)

    s = np.rint(s)
    s = np.clip(s, 0, 255)
    return s.astype(np.uint8)




def get_gamma_transform(gamma: float) -> np.ndarray:
    
    gamma = float(gamma)
    r = np.arange(256, dtype=np.float64)
    # r/255 in [0,1]
    x = r / 255.0
    s = 255.0 * np.power(x, gamma)
    return _to_uint8_lut(s)


def get_hist_equalize_transform(image: np.ndarray, do_stretching: bool) -> np.ndarray:
  
    img = _ensure_gray_uint8(image)
    if img is None:
        return np.arange(256, dtype=np.uint8)

    # Histogram and CDF
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return np.arange(256, dtype=np.uint8)

    cdf = np.cumsum(hist) / total  # in [0,1]

    if do_stretching:
        # Standard "stretched equalization": normalize by first nonzero CDF
        nonzero = np.nonzero(hist)[0]
        if nonzero.size == 0:
            cdf_min = 0.0
        else:
            cdf_min = float(cdf[nonzero[0]])
        denom = (1.0 - cdf_min)
        if denom <= 0:
            s = np.zeros(256, dtype=np.float64)
        else:
            s = (cdf - cdf_min) / denom * 255.0
    else:
        s = cdf * 255.0

    return _to_uint8_lut(s)


def get_piecewise_linear_transform(points: list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    
    if points is None or len(points) == 0:
        return np.arange(256, dtype=np.uint8)

    # Normalize to list of tuples
    pts = [(float(p[0]), float(p[1])) for p in points]
    pts.sort(key=lambda t: t[0])

    r_pts, s_pts = zip(*pts)
    r_pts = np.array(r_pts, dtype=np.float64)
    s_pts = np.array(s_pts, dtype=np.float64)

    # Interpolate at every integer r in [0,255]
    r = np.arange(256, dtype=np.float64)
    s = np.interp(r, r_pts, s_pts, left=s_pts[0], right=s_pts[-1])

    return _to_uint8_lut(s)




# ----------------------------
# Gradio UI (main)
# ----------------------------

def main() -> None:
    def _run(image, mode, gamma, max_r):
        img = _ensure_gray_uint8(image)
        if img is None:
            return None

        mode = str(mode)

        if mode == "Histogram equalization":
            lut = get_hist_equalize_transform(img, do_stretching=False)
        elif mode == "Histogram equalization (stretched)":
            lut = get_hist_equalize_transform(img, do_stretching=True)
        elif mode == "Gamma / power-law":
            lut = get_gamma_transform(gamma)
        elif mode == "Log":
            lut = get_log_transform(int(max_r))
        else:
            lut = np.arange(256, dtype=np.uint8)

        out = apply_intensity_transform(img, lut)
        return out

    with gr.Blocks(title="A01 Intensity Transforms") as demo:
        gr.Markdown("# A01: Intensity transformations (LUT-based)")

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input image (grayscale)", type="numpy", image_mode="L")
                mode = gr.Radio(
                    choices=[
                        "Histogram equalization",
                        "Histogram equalization (stretched)",
                        "Gamma / power-law",
                        "Log",
                    ],
                    value="Histogram equalization",
                    label="Transform",
                )
                gamma = gr.Slider(0.1, 5.0, value=1.0, step=0.05, label="Gamma (used for Gamma / power-law)")
                max_r = gr.Slider(1, 255, value=255, step=1, label="max_r (used for Log)")
                run_btn = gr.Button("Apply transform")

            with gr.Column(scale=1):
                out = gr.Image(label="Output image", type="numpy", image_mode="L")

        # Run on button click and also when controls change (if image is present)
        run_btn.click(_run, inputs=[inp, mode, gamma, max_r], outputs=[out])
        mode.change(_run, inputs=[inp, mode, gamma, max_r], outputs=[out])
        gamma.change(_run, inputs=[inp, mode, gamma, max_r], outputs=[out])
        max_r.change(_run, inputs=[inp, mode, gamma, max_r], outputs=[out])
        inp.change(_run, inputs=[inp, mode, gamma, max_r], outputs=[out])

    demo.launch()


if __name__ == "__main__":
    main()