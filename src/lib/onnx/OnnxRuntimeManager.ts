import * as ort from "onnxruntime-web";

export type ExecutionProvider = "webgpu" | "cpu";

export interface SessionOptions {
  readonly executionProviders?: ExecutionProvider[];
}

/**
 * Singleton responsible for configuring the ONNX Runtime environment and creating inference sessions.
 */
export class OnnxRuntimeManager {
  private static instance: OnnxRuntimeManager | null = null;

  private constructor() {
    this.configureEnvironment();
  }

  public static getInstance(): OnnxRuntimeManager {
    if (!OnnxRuntimeManager.instance) {
      OnnxRuntimeManager.instance = new OnnxRuntimeManager();
    }
    return OnnxRuntimeManager.instance;
  }

  public async createSession(modelUrl: string, options: SessionOptions = {}): Promise<ort.InferenceSession> {
    const providers = options.executionProviders ?? ["cpu"];

    for (const provider of providers) {
      try {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
        });
        return session;
      } catch (error) {
        console.error(`Failed to create session with provider ${provider} for ${modelUrl}`, error);
      }
    }

    throw new Error(`Unable to create inference session for ${modelUrl}`);
  }

  public get runtime(): typeof ort {
    return ort;
  }

  private configureEnvironment(): void {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
    ort.env.logLevel = "warning";
  }
}

export { ort };
