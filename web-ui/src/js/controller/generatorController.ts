import { getInputVectorAsRGB2DImage } from "../misc";
import { ModelController} from "./modelController";
import ApiClient from "../apiClient";
import { ImageRenderer } from "../renderer/imageRenderer";
import WebUI from "../webUI";
import {number3DMatrix, RGB2DImage} from "../types/types";

export default class GeneratorController extends ModelController {
    private rendererInput: ImageRenderer;
    private rendererOutput: ImageRenderer;
    private choiceLayerGenerator: HTMLSelectElement;

    constructor(
        callingWebUI: WebUI,
        apiClient: ApiClient,
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
        console.log('===> Refreshing Input Generator')
        const latentVectorAsRGB2DImage = getInputVectorAsRGB2DImage(
            this.callingWebUI.getLatentVector(),
            this.callingWebUI.latentSpaceSizeSqrt,
            this.callingWebUI.maxValueVisualizationInput,
        );
        this.rendererInput.changeImage(latentVectorAsRGB2DImage);

        this.inputData = [[this.callingWebUI.getLatentVector()]];
         // why call it twice todo fix getLatentVector()
        console.log('===> Refreshing Inside Generator')
        await this.refreshInside(this.choiceLayerGenerator.value);

        console.log('===> Refreshing Output Generator')
        const dataGenerator: number3DMatrix = (await this.apiClient.getModelPrediction(
            this.inputData,
            "generator",
            this.lastLayerName,
        )) ;
        this.rendererOutput.changeImage(dataGenerator as RGB2DImage);

        await this.callingWebUI.updateDiscriminator(dataGenerator);
    }
}