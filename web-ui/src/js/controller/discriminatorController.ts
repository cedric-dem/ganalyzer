import {toPercentage} from "../misc.js";
import {ApiClient, ModelController, WebUICallingContext} from "./modelController.ts";
import {ImageRenderer} from "../renderer/imageRenderer.js";

type DiscriminatorWebUIContext = WebUICallingContext & {
    imageSize: number;
};

export default class DiscriminatorController extends ModelController {
    private rendererInput: ImageRenderer;
    private layerChoiceInsideVisualization: HTMLSelectElement;
    private predictionOutputText: HTMLElement;

    constructor(
        callingWebUI: DiscriminatorWebUIContext,
        apiClient: ApiClient,
        layerLocation: string,
    ) {
        super(
            callingWebUI,
            "discriminator",
            apiClient,
            "div_visualization_inside_discriminator",
            "labelDiscriminatorEpochValue",
            layerLocation,
        );

        this.rendererInput = new ImageRenderer("div_visualization_input_discriminator");

        this.layerChoiceInsideVisualization = document.getElementById(
            "choice_layer_discriminator",
        ) as HTMLSelectElement;
        this.predictionOutputText = document.getElementById("prediction_output_text") as HTMLElement;
    }

    initialize(): void {
        this.rendererInput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    changeInputImage(generatorImage: number[][]): void {
        this.inputData = generatorImage;
    }

    async refreshAll(): Promise<void> {
        const resultDiscriminator = (await this.apiClient.getModelPrediction(
            this.inputData,
            "discriminator",
            this.lastLayerName ?? "",
        )) as number;

        this.rendererInput.changeImage(this.inputData as number[][]);

        await this.refreshInside(this.layerChoiceInsideVisualization.value);

        this.predictionOutputText.textContent = this.getTextPrediction(resultDiscriminator);
    }

    private getTextPrediction(score: number): string {
        const label = score > 0.5 ? "real" : "fake";
        return `${label} image (${toPercentage(score)})`;
    }
}