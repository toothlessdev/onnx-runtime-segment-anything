import { OnnxRuntimeManager } from "../lib/onnx/OnnxRuntimeManager";
import type { InferenceSession, Tensor } from "onnxruntime-web";

export interface Sam2EncodedFeatures {
  readonly highResFeats0: Tensor;
  readonly highResFeats1: Tensor;
  readonly imageEmbed: Tensor;
}

export class Sam2Encoder {
  private session: InferenceSession | null = null;

  public constructor(
    private readonly runtimeManager: OnnxRuntimeManager = OnnxRuntimeManager.getInstance(),
  ) {}

  public get isLoaded(): boolean {
    return this.session !== null;
  }

  public async load(modelUrl: string): Promise<void> {
    this.session = await this.runtimeManager.createSession(modelUrl, {
      executionProviders: ["cpu"],
    });
  }

  public async encode(imageTensor: Tensor): Promise<Sam2EncodedFeatures> {
    if (!this.session) {
      throw new Error("Encoder session has not been loaded");
    }

    const outputs = await this.session.run({ image: imageTensor });
    const names = this.session.outputNames;

    const resolve = (expected: string, fallbackIndex: number) => {
      const name = names.find((candidate) => candidate.includes(expected)) ?? names[fallbackIndex];
      return outputs[name] as Tensor;
    };

    return {
      highResFeats0: resolve("high_res_feats_0", 0),
      highResFeats1: resolve("high_res_feats_1", 1),
      imageEmbed: resolve("image_embed", names.length - 1),
    };
  }
}
