import cv2
import gradio as gr
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def read_kernel_file(filepath):

    with open(filepath, "r") as f:
        line = f.readline().strip()

    tokens = line.split()
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])

    kernel = np.array(tokens[2:], dtype=np.float64).reshape((rowCnt, colCnt))
    return kernel


def _finalize_output(output, alpha=1.0, beta=0.0, convert_uint8=True):

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
    return output


def _same_padding_sizes(kernel_shape):

    kh, kw = kernel_shape
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def do_convolution_slow(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kh, kw = kernel.shape
    ih, iw = image.shape

    pad_top, pad_bottom, pad_left, pad_right = _same_padding_sizes(kernel.shape)

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )

    flipped_kernel = np.flip(kernel)
    output = np.zeros((ih, iw), dtype=np.float64)

    for r in range(ih):
        for c in range(iw):
            total = 0.0
            for kr in range(kh):
                for kc in range(kw):
                    total += padded[r + kr, c + kc] * flipped_kernel[kr, kc]
            output[r, c] = total

    return _finalize_output(output, alpha=alpha, beta=beta, convert_uint8=convert_uint8)


def do_convolution_fast(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    pad_top, pad_bottom, pad_left, pad_right = _same_padding_sizes(kernel.shape)

    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    )

    flipped_kernel = np.flip(kernel)

    windows = sliding_window_view(padded, kernel.shape)
    output = np.sum(windows * flipped_kernel, axis=(2, 3))

    return _finalize_output(output, alpha=alpha, beta=beta, convert_uint8=convert_uint8)


def do_convolution_fourier(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    """
    Convolution using the Fourier transform.
    Uses cv2.getOptimalDFTSize() for efficiency.
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    ih, iw = image.shape
    kh, kw = kernel.shape

    full_h = ih + kh - 1
    full_w = iw + kw - 1

    dft_h = cv2.getOptimalDFTSize(full_h)
    dft_w = cv2.getOptimalDFTSize(full_w)

    image_pad = np.zeros((dft_h, dft_w), dtype=np.float64)
    kernel_pad = np.zeros((dft_h, dft_w), dtype=np.float64)

    image_pad[:ih, :iw] = image
    kernel_pad[:kh, :kw] = kernel

    image_fft = np.fft.fft2(image_pad)
    kernel_fft = np.fft.fft2(kernel_pad)

    conv_full = np.fft.ifft2(image_fft * kernel_fft).real
    conv_full = conv_full[:full_h, :full_w]

    pad_top, _, pad_left, _ = _same_padding_sizes(kernel.shape)
    output = conv_full[pad_top:pad_top + ih, pad_left:pad_left + iw]

    return _finalize_output(output, alpha=alpha, beta=beta, convert_uint8=convert_uint8)


def _get_uploaded_file_path(file_obj):
    """
    Gradio file inputs may come in as a path string or an object with a .name.
    """
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    return None


def _filter_image(image, kernel_file, method, alpha, beta):
    """
    Callback for Gradio.
    """
    if image is None:
        raise gr.Error("Please load a grayscale image.")

    filepath = _get_uploaded_file_path(kernel_file)
    if filepath is None:
        raise gr.Error("Please load a kernel file.")

    kernel = read_kernel_file(filepath)

    if method == "Fast":
        return do_convolution_fast(image, kernel, alpha=alpha, beta=beta, convert_uint8=True)
    elif method == "Fourier":
        return do_convolution_fourier(image, kernel, alpha=alpha, beta=beta, convert_uint8=True)
    else:
        raise gr.Error("Unknown filtering method selected.")


def main():
    with gr.Blocks(title="A02 Convolution Demo") as demo:
        gr.Markdown("## A02 Convolution Demo")
        gr.Markdown("Load a grayscale image and a kernel file, then apply convolution.")

        with gr.Row():
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                image_mode="L"
            )

            output_image = gr.Image(
                label="Filtered Image",
                type="numpy",
                image_mode="L"
            )

        with gr.Row():
            kernel_file = gr.File(label="Kernel File")
            method = gr.Radio(
                choices=["Fast", "Fourier"],
                value="Fast",
                label="Method"
            )

        with gr.Row():
            alpha = gr.Number(value=1.0, label="Alpha")
            beta = gr.Number(value=0.0, label="Beta")

        apply_button = gr.Button("Filter Image")

        apply_button.click(
            fn=_filter_image,
            inputs=[input_image, kernel_file, method, alpha, beta],
            outputs=output_image
        )

    demo.launch()


if __name__ == "__main__":
    main()