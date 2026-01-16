import {getInputVectorAsMatrix} from "../misc.js";
import {ModelController} from "./modelController.js";
import {ImageRenderer} from "../renderer/imageRenderer.js";

export default class GeneratorController extends ModelController {
    constructor(callingWebUI, apiClient, latentSpaceSize, layer_location) {
        super(callingWebUI, "generator", apiClient, "div_visualization_inside_generator", "labelGeneratorEpochValue", layer_location);

        this.rendererInput = new ImageRenderer("div_visualization_input_generator");
        this.rendererOutput = new ImageRenderer("div_visualization_output_generator");

        this.choiceLayerGenerator = document.getElementById("choice_layer_generator")
    }

    initialize() {
        this.rendererInput.initializeImage(this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.latentSpaceSizeSqrt);
        this.rendererOutput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    async refreshAll() {
        // input
        const latentVectorAsMatrix = getInputVectorAsMatrix(this.callingWebUI.getLatentVector(), this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.maxValueVisualizationInput);
        this.rendererInput.changeImage(latentVectorAsMatrix);

        //inside
        this.inputData = this.callingWebUI.getLatentVector()
        await this.refreshInside(this.choiceLayerGenerator.value);

        //output
        const dataGenerator = await this.apiClient.getModelPrediction(this.callingWebUI.getLatentVector(), "generator", this.lastLayerName);
        this.rendererOutput.changeImage(dataGenerator);

        // update discriminator
        await this.callingWebUI.updateDiscriminator(dataGenerator);
    }
}