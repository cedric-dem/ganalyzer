import {convertNumberToStringPercentage} from "../misc";
import {ModelController} from "./modelController";
import ApiClient from "../apiClient";
import {ImageRenderer} from "../renderer/imageRenderer";
import WebUI from "../webUI";
import {number3DMatrix, RGB2DImage} from "../types/types";

export default class DiscriminatorController extends ModelController {
    private rendererInput: ImageRenderer;
    private layerChoiceInsideVisualization: HTMLSelectElement;
    private predictionOutputText: HTMLElement;

    constructor(
        callingWebUI: WebUI,
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

    changeInputImage(generatorImage: number3DMatrix): void {
        this.inputData = generatorImage;
    }

    async refreshAll(): Promise<void> {
        console.log('===> Refreshing Input Discriminator')
        this.rendererInput.changeImage(this.inputData as RGB2DImage);

        console.log('===> Refreshing Inside Discriminator')
        await this.refreshInside(this.layerChoiceInsideVisualization.value);

        console.log('===> Refreshing Output Discriminator')
        const resultDiscriminator: number3DMatrix = (await this.apiClient.getModelPrediction(
            this.inputData,
            "discriminator",
            this.lastLayerName,
        ));

        this.predictionOutputText.textContent = this.getTextPrediction(resultDiscriminator[0][0][0]);
    }

    private getTextPrediction(score: number): string {
        const label = score > 0.5 ? "real" : "fake";
        return `${label} image (${convertNumberToStringPercentage(score)})`;
    }
}