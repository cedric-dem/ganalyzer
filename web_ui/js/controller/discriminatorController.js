import {toPercentage} from "../misc.js";
import {ModelController} from "./modelController.js";
import {ImageRenderer} from "../renderer/imageRenderer.js";

class DiscriminatorController extends ModelController {
    constructor(callingWebUI, apiClient, layer_location) {
        super(callingWebUI, "discriminator", apiClient, "div_visualization_inside_discriminator", "labelDiscriminatorEpochValue", layer_location);

        this.rendererInput = new ImageRenderer("div_visualization_input_discriminator");

        this.layerChoiceInsideVisualization = document.getElementById("choice_layer_discriminator")
        this.predictionOutputText = document.getElementById("prediction_output_text")
    }

    initialize() {
        this.rendererInput.initializeImage(this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    changeInputImage(generatorImage) {
        this.inputData = generatorImage;
    }

    async refreshAll() {
        const resultDiscriminator = await this.apiClient.getModelPrediction(this.inputData, "discriminator", this.lastLayerName);

        //change input
        this.rendererInput.changeImage(this.inputData);

        //change inside
        await this.refreshInside(this.layerChoiceInsideVisualization.value)

        // change output
        this.predictionOutputText.textContent = this.getTextPrediction(resultDiscriminator);
    }

    getTextPrediction(score) {
        const label = score > 0.5 ? "real" : "fake";
        return `${label} image (${toPercentage(score)})`;
    }

}

export default DiscriminatorController;