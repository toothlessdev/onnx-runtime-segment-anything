import { OnnxRuntimeManager } from "../lib/onnx/OnnxRuntimeManager";
import type { Sam2EncodedFeatures } from "./Sam2Encoder";
import type { InferenceSession, Tensor } from "onnxruntime-web";

export interface Sam2DecoderRequest extends Sam2EncodedFeatures {
  readonly pointCoords: Tensor;
  readonly pointLabels: Tensor;
  readonly maskInput?: Tensor;
  readonly hasMaskInput?: Tensor;
}

export interface Sam2DecoderResult {
  readonly masks: Tensor;
  readonly iouPredictions: Tensor;
}

export class Sam2Decoder {
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

  public async decode(request: Sam2DecoderRequest): Promise<Sam2DecoderResult> {
    if (!this.session) {
      throw new Error("Decoder session has not been loaded");
    }

    const inputs: Record<string, Tensor> = {};
    const names = this.session.inputNames;

    const assignIfExpected = (name: string, tensor?: Tensor) => {
      if (tensor && names.includes(name)) {
        inputs[name] = tensor;
      }
    };

    assignIfExpected("image_embed", request.imageEmbed);
    assignIfExpected("high_res_feats_0", request.highResFeats0);
    assignIfExpected("high_res_feats_1", request.highResFeats1);
    assignIfExpected("point_coords", request.pointCoords);
    assignIfExpected("point_labels", request.pointLabels);
    assignIfExpected("mask_input", request.maskInput);
    assignIfExpected("has_mask_input", request.hasMaskInput);

    const outputs = await this.session.run(inputs);

    const masks = this.extractOutput(outputs, "masks");
    const iouPredictions = this.extractOutput(outputs, "iou_predictions");

    return { masks, iouPredictions };
  }

  private extractOutput(outputs: Record<string, Tensor>, expectedName: string): Tensor {
    if (!this.session) {
      throw new Error("Decoder session is not available");
    }

    const name = this.session.outputNames.find((candidate) => candidate.includes(expectedName))
      ?? this.session.outputNames[0];
    return outputs[name];
  }
}
