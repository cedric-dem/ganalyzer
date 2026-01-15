import {toPercentage} from "./misc.js";
import {ModelController} from "./ModelController.js";

class DiscriminatorController extends ModelController {
    constructor(callingWebUI, apiClient, imageGridRenderer) {
        super(callingWebUI, "discriminator", apiClient, "div_visualization_inside_discriminator", "labelDiscriminatorEpochValue");

        this.imageGridRenderer = imageGridRenderer;

        this.discriminatorInputImagePixels = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage("div_visualization_input_discriminator", this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    changeInputImage(generatorImage) {
        this.inputData = generatorImage;
    }

    async refreshAll() {
        const resultDiscriminator = await this.apiClient.getModelPrediction(this.inputData, "discriminator", this.lastLayerName);

        //change input
        this.imageGridRenderer.changeImage(this.inputData, this.discriminatorInputImagePixels);

        //change inside
        await this.refreshInside(document.getElementById("choice_layer_discriminator").value)

        // change output
        document.getElementById("prediction_output_text").textContent = this.getTextPrediction(resultDiscriminator);
    }

    getTextPrediction(score) {
        const label = score > 0.5 ? "real" : "fake";
        return `${label} image (${toPercentage(score)})`;
    }

}

export default DiscriminatorController;