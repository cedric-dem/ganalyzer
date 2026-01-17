import { getInputVectorAsRGB2DImage } from "../misc";
import { ModelController} from "./modelController";
import ApiClient from "../apiClient";
import { ImageRenderer } from "../renderer/imageRenderer";
import WebUI from "../webUI";

export default class GeneratorController extends ModelController {
    private rendererInput: ImageRenderer;
    private rendererOutput: ImageRenderer;
    private choiceLayerGenerator: HTMLSelectElement;

    constructor(
        callingWebUI: WebUI,
        apiClient: ApiClient,
        latentSpaceSize: number,
        layerLocation: string,
    ) {
        super(
            callingWebUI,
            "generator",
            apiClient,
            "div_visualization_inside_generator",
            "labelGeneratorEpochValue",
            layerLocation,
        );

        this.rendererInput = new ImageRenderer("div_visualization_input_generator");
        this.rendererOutput = new ImageRenderer("div_visualization_output_generator");

        this.choiceLayerGenerator = document.getElementById("choice_layer_generator") as HTMLSelectElement;
    }

    initialize(): void {
        this.rendererInput.initializeImage(
            this.callingWebUI.latentSpaceSizeSqrt,
            this.callingWebUI.latentSpaceSizeSqrt,
        );
        this.rendererOutput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    async refreshAll(): Promise<void> {
        const latentVectorAsRGB2DImage = getInputVectorAsRGB2DImage(
            this.callingWebUI.getLatentVector(),
            this.callingWebUI.latentSpaceSizeSqrt,
            this.callingWebUI.maxValueVisualizationInput,
        );
        this.rendererInput.changeImage(latentVectorAsRGB2DImage);

        this.inputData = this.callingWebUI.getLatentVector();
        await this.refreshInside(this.choiceLayerGenerator.value);

        const dataGenerator = (await this.apiClient.getModelPrediction(
            this.callingWebUI.getLatentVector(),
            "generator",
            this.lastLayerName,
        )) as number[][];
        this.rendererOutput.changeImage(dataGenerator);

        await this.callingWebUI.updateDiscriminator(dataGenerator);
    }
}