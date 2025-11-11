import type { Tensor } from "onnxruntime-web";

export class MaskPostProcessor {
  public static selectBestMask(ious: Tensor): number {
    const data = ious.data as Float32Array;
    let bestIndex = 0;
    let bestValue = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < data.length; i++) {
      if (data[i] > bestValue) {
        bestIndex = i;
        bestValue = data[i];
      }
    }

    return bestIndex;
  }

  public static maskToCanvas(masks: Tensor, maskIndex: number): HTMLCanvasElement {
    const [, , height, width] = masks.dims;
    const stride = width * height;
    const offset = maskIndex * stride;
    const data = (masks.data as Float32Array).subarray(offset, offset + stride);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("Failed to acquire 2D context for mask canvas");
    }

    const rgba = new Uint8ClampedArray(stride * 4);

    for (let i = 0; i < stride; i++) {
      const alpha = data[i] > 0 ? 200 : 0;
      const pixelIndex = i * 4;
      rgba[pixelIndex] = 255;
      rgba[pixelIndex + 1] = 0;
      rgba[pixelIndex + 2] = 0;
      rgba[pixelIndex + 3] = alpha;
    }

    const imageData = new ImageData(rgba, width, height);
    context.putImageData(imageData, 0, 0);

    return canvas;
  }
}
