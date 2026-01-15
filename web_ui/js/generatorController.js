import {getInputVectorAsMatrix, getRandomNormalFloat} from "./misc.js";
import {ModelController} from "./ModelController.js";
import {ImageGridRenderer} from "./imageRenderer.js";

class GeneratorController extends ModelController {
    constructor(callingWebUI, apiClient, sliderGridRenderer, discriminatorController, latentSpaceSize) {
        super(callingWebUI, "generator", apiClient, "div_visualization_inside_generator", "labelGeneratorEpochValue");

        this.inputData = new Array(latentSpaceSize).fill(0);

        this.sliderGridRenderer = sliderGridRenderer;

        this.discriminatorController = discriminatorController;

        this.slidersGrid = null;

        this.rendererInput = new ImageGridRenderer("div_visualization_input_generator");
        this.rendererOutput = new ImageGridRenderer("div_visualization_output_generator");
    }

    initialize() {
        this.rendererInput.initializeImage( this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.latentSpaceSizeSqrt);
        this.rendererOutput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
        this.sliderGridRenderer.initializeGeneratorSliders(this.callingWebUI.latentSpaceSizeSqrt, (i, j, newValue) => this.handleSliderValueChange(i, j, newValue));
    }

    async refreshAll() {
        // input
        const latentVectorAsMatrix = getInputVectorAsMatrix(this.inputData, this.callingWebUI.latentSpaceSizeSqrt, this.callingWebUI.maxValueVisualizationInput);
        this.rendererInput.changeImage(latentVectorAsMatrix);

        //inside
        await this.refreshInside(document.getElementById("choice_layer_generator").value);

        //output
        const dataGenerator = await this.apiClient.getModelPrediction(this.inputData, "generator", this.lastLayerName);
        this.rendererOutput.changeImage(dataGenerator);

        // update discriminator
        this.discriminatorController.changeInputImage(dataGenerator);
        await this.discriminatorController.refreshAll();
    }

    randomizeInput() {
        const mu = parseFloat(document.getElementById("sliderMuValue").value);
        const sigma = parseFloat(document.getElementById("sliderSigmaValue").value);

        for (let i = 0; i < this.callingWebUI.latentSpaceSize; i++) {
            this.inputData[i] = getRandomNormalFloat(mu, sigma);
        }
        this.refreshSliders();
        this.refreshAll();
    }

    setConstantInput() {
        const k = parseFloat(document.getElementById("sliderConstantValue").value);
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