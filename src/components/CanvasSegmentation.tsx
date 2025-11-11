import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { OnnxRuntimeManager } from "../lib/onnx/OnnxRuntimeManager";
import { ImagePreprocessor } from "../lib/preprocessing/ImagePreprocessor";
import { MaskPostProcessor } from "../lib/postprocessing/MaskPostProcessor";
import { Sam2Encoder } from "../models/Sam2Encoder";
import type { Sam2EncodedFeatures } from "../models/Sam2Encoder";
import { Sam2Decoder } from "../models/Sam2Decoder";
import { SAM2_DECODER_URL, SAM2_ENCODER_URL } from "../models/constants";

interface CanvasSegmentationProps {
  readonly imageSrc: string;
}

interface ViewSize {
  width: number;
  height: number;
}

export const CanvasSegmentation: React.FC<CanvasSegmentationProps> = ({ imageSrc }) => {
  const runtimeManager = useMemo(() => OnnxRuntimeManager.getInstance(), []);
  const runtime = runtimeManager.runtime;
  const encoder = useMemo(() => new Sam2Encoder(runtimeManager), [runtimeManager]);
  const decoder = useMemo(() => new Sam2Decoder(runtimeManager), [runtimeManager]);

  const viewCanvasRef = useRef<HTMLCanvasElement>(null);
  const preprocessedCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [encodedFeatures, setEncodedFeatures] = useState<Sam2EncodedFeatures | null>(null);
  const [status, setStatus] = useState<string>("Initializing models...");
  const [error, setError] = useState<string | null>(null);
  const [modelsReady, setModelsReady] = useState<boolean>(false);
  const viewSizeRef = useRef<ViewSize>({ width: 0, height: 0 });
  const baseImageRef = useRef<HTMLImageElement | null>(null);

  // Load encoder and decoder sessions once.
  useEffect(() => {
    let cancelled = false;

    setModelsReady(false);

    const loadModels = async () => {
      try {
        setStatus("Loading decoder...");
        if (!decoder.isLoaded) {
          await decoder.load(SAM2_DECODER_URL);
        }

        setStatus("Loading encoder...");
        if (!encoder.isLoaded) {
          await encoder.load(SAM2_ENCODER_URL);
        }

        if (!cancelled) {
          setStatus("Models ready");
          setModelsReady(true);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setModelsReady(false);
        }
      }
    };

    loadModels();

    return () => {
      cancelled = true;
    };
  }, [decoder, encoder]);

  const drawBaseImage = useCallback((image: HTMLImageElement) => {
    const canvas = viewCanvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const maxWidth = 720;
    const ratio = image.width / image.height;
    const width = Math.min(maxWidth, image.width);
    const height = Math.round(width / ratio);

    canvas.width = width;
    canvas.height = height;

    context.clearRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);

    viewSizeRef.current = { width, height };
  }, []);

  const encodeImage = useCallback(async () => {
    if (!encoder.isLoaded || !preprocessedCanvasRef.current) {
      return;
    }

    try {
      setStatus("Encoding image...");
      const tensor = ImagePreprocessor.canvasToCHWTensor(preprocessedCanvasRef.current);
      const features = await encoder.encode(tensor);
      setEncodedFeatures(features);
      setStatus("Ready");
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [encoder]);

  // Load image whenever the source changes.
  useEffect(() => {
    if (!imageSrc) return;
    setError(null);
    setEncodedFeatures(null);

    let cancelled = false;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = imageSrc;

    setStatus("Loading image...");

    image.onload = () => {
      if (cancelled) return;
      baseImageRef.current = image;
      drawBaseImage(image);
      const { canvas } = ImagePreprocessor.drawToSquare1024(image);
      preprocessedCanvasRef.current = canvas;
      void encodeImage();
    };

    image.onerror = () => {
      if (cancelled) return;
      setError("Failed to load image");
    };

    return () => {
      cancelled = true;
    };
  }, [drawBaseImage, encodeImage, imageSrc]);

  useEffect(() => {
    if (!modelsReady) {
      return;
    }

    void encodeImage();
  }, [modelsReady, encodeImage]);

  const restoreBaseImage = useCallback(() => {
    const canvas = viewCanvasRef.current;
    const image = baseImageRef.current;
    if (!canvas || !image) return;
    drawBaseImage(image);
  }, [drawBaseImage]);

  const handleClick = useCallback(
    async (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!decoder.isLoaded || !encodedFeatures || !preprocessedCanvasRef.current) {
        return;
      }

      const canvas = viewCanvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const { width, height } = viewSizeRef.current;

      const relativeX = ((event.clientX - rect.left) / rect.width) * width;
      const relativeY = ((event.clientY - rect.top) / rect.height) * height;

      const normalizedX = (relativeX / width) * 1024;
      const normalizedY = (relativeY / height) * 1024;

      const labelValue = event.shiftKey ? 0 : 1;

      setError(null);

      try {
        const pointCoords = new runtime.Tensor("float32", new Float32Array([normalizedX, normalizedY]), [1, 1, 2]);
        const pointLabels = new runtime.Tensor("float32", new Float32Array([labelValue]), [1, 1]);
        const maskInput = new runtime.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);
        const hasMaskInput = new runtime.Tensor("float32", new Float32Array([0]), [1]);

        const result = await decoder.decode({
          highResFeats0: encodedFeatures.highResFeats0,
          highResFeats1: encodedFeatures.highResFeats1,
          imageEmbed: encodedFeatures.imageEmbed,
          pointCoords,
          pointLabels,
          maskInput,
          hasMaskInput,
        });

        const maskIndex = MaskPostProcessor.selectBestMask(result.iouPredictions);
        const maskCanvas = MaskPostProcessor.maskToCanvas(result.masks, maskIndex);

        restoreBaseImage();
        const context = canvas.getContext("2d");
        if (!context) return;
        context.save();
        context.globalAlpha = 0.45;
        context.drawImage(maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height, 0, 0, width, height);
        context.restore();
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [decoder, encodedFeatures, restoreBaseImage, runtime, setError],
  );

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <canvas ref={viewCanvasRef} onClick={handleClick} style={{ width: "100%", maxWidth: 720, cursor: "crosshair" }} />
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      {!error && status && <p style={{ color: "#4b5563" }}>{status}</p>}
    </div>
  );
};
