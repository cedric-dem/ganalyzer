import {get1DNullArray, getInputVectorAsMatrix, getRandomNormalFloat} from "../misc.js";
import {ModelController} from "./modelController.js";
import {ImageRenderer} from "../renderer/imageRenderer.js";

class GeneratorController extends ModelController {
    constructor(callingWebUI, apiClient, sliderGridRenderer, latentSpaceSize, layer_location) {
        super(callingWebUI, "generator", apiClient, "div_visualization_inside_generator", "labelGeneratorEpochValue", layer_location);

        this.inputData = get1DNullArray(latentSpaceSize);

        this.sliderGridRenderer = sliderGridRenderer;

        this.slidersGrid = null;

        this.rendererInput = new ImageRenderer("div_visualization_input_generator");
        this.rendererOutput = new ImageRenderer("div_visualization_output_generator");

        this.choiceLayerGenerator = document.getElementById("choice_layer_generator")
    }

    initialize() {
        this.rendererInput.initializeImage(this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.latentSpaceSizeSqrt);
        this.rendererOutput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
        this.sliderGridRenderer.initializeGeneratorSliders(this.callingWebUI.latentSpaceSizeSqrt, (i, j, newValue) => this.handleSliderValueChange(i, j, newValue));
    }

    async refreshAll() {
        // input
        const latentVectorAsMatrix = getInputVectorAsMatrix(this.inputData, this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.maxValueVisualizationInput);
        this.rendererInput.changeImage(latentVectorAsMatrix);

        //inside
        await this.refreshInside(this.choiceLayerGenerator.value);

        //output
        const dataGenerator = await this.apiClient.getModelPrediction(this.inputData, "generator", this.lastLayerName);
        this.rendererOutput.changeImage(dataGenerator);

        // update discriminator
        await this.callingWebUI.updateDiscriminator(dataGenerator);
    }

    randomizeInput() {
        const mu = this.callingWebUI.getMuValue();
        const sigma = this.callingWebUI.getSigmaValue();

        for (let i = 0; i < this.callingWebUI.latentSpaceSize; i++) {
            this.inputData[i] = getRandomNormalFloat(mu, sigma);
        }
        this.refreshSliders();
        this.refreshAll();
    }

    setConstantInput() {
        const k = this.callingWebUI.getKValue();
        for (let i = 0; i < this.inputData.length; i++) {
            this.inputData[i] = k;
        }
        this.refreshSliders();
        this.refreshAll();
    }

    refreshSliders() {
        this.sliderGridRenderer.refreshSliders(this.inputData, this.callingWebUI.latentSpaceSizeSqrt);
    }

    handleSliderValueChange(i, j, newValue) {
        this.inputData[i * this.callingWebUI.latentSpaceSizeSqrt + j] = newValue;
        this.refreshAll();
    }
}

export default GeneratorController;