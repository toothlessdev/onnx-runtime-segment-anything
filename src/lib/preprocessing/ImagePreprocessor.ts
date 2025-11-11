import { OnnxRuntimeManager } from "../onnx/OnnxRuntimeManager";
import type { Tensor } from "onnxruntime-web";

export interface SquareCanvasResult {
  readonly canvas: HTMLCanvasElement;
  readonly scale: number;
  readonly padding: { x: number; y: number };
}

export class ImagePreprocessor {
  /**
   * Resizes an image into a square canvas, padding the shorter side before scaling to 1024x1024.
   */
  public static drawToSquare1024(image: HTMLImageElement): SquareCanvasResult {
    const maxDimension = Math.max(image.width, image.height);
    const squareCanvas = document.createElement("canvas");
    squareCanvas.width = maxDimension;
    squareCanvas.height = maxDimension;
    const squareContext = squareCanvas.getContext("2d");

    if (!squareContext) {
      throw new Error("Failed to acquire 2D context for square canvas");
    }

    const padX = image.height > image.width ? Math.floor((maxDimension - image.width) / 2) : 0;
    const padY = image.width > image.height ? Math.floor((maxDimension - image.height) / 2) : 0;

    squareContext.drawImage(
      image,
      0,
      0,
      image.width,
      image.height,
      padX,
      padY,
      maxDimension - padX * 2,
      maxDimension - padY * 2,
    );

    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = 1024;
    outputCanvas.height = 1024;
    const outputContext = outputCanvas.getContext("2d");

    if (!outputContext) {
      throw new Error("Failed to acquire 2D context for output canvas");
    }

    outputContext.drawImage(squareCanvas, 0, 0, maxDimension, maxDimension, 0, 0, 1024, 1024);

    return {
      canvas: outputCanvas,
      scale: 1024 / maxDimension,
      padding: { x: padX, y: padY },
    };
  }

  /**
   * Converts a 1024x1024 canvas into a CHW tensor normalized to 0-1 values.
   */
  public static canvasToCHWTensor(canvas: HTMLCanvasElement): Tensor {
    const width = canvas.width;
    const height = canvas.height;
    const context = canvas.getContext("2d");

    if (!context) {
      throw new Error("Failed to acquire 2D context for tensor conversion");
    }

    const { data } = context.getImageData(0, 0, width, height);
    const pixelCount = width * height;
    const red = new Float32Array(pixelCount);
    const green = new Float32Array(pixelCount);
    const blue = new Float32Array(pixelCount);

    for (let i = 0, p = 0; i < data.length; i += 4, p++) {
      red[p] = data[i] / 255;
      green[p] = data[i + 1] / 255;
      blue[p] = data[i + 2] / 255;
    }

    const chw = new Float32Array(pixelCount * 3);
    chw.set(red, 0);
    chw.set(green, pixelCount);
    chw.set(blue, pixelCount * 2);

    const runtime = OnnxRuntimeManager.getInstance().runtime;

    return new runtime.Tensor("float32", chw, [1, 3, width, height]);
  }
}
