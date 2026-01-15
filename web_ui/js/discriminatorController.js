import {toPercentage} from "./misc.js";
import {ModelController} from "./ModelController.js";
import {ImageGridRenderer} from "./imageRenderer.js";

class DiscriminatorController extends ModelController {
    constructor(callingWebUI, apiClient) {
        super(callingWebUI, "discriminator", apiClient, "div_visualization_inside_discriminator", "labelDiscriminatorEpochValue");

        this.rendererInput = new ImageGridRenderer("div_visualization_input_discriminator");
    }

    initialize() {
        this.rendererInput.initializeImage( this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    changeInputImage(generatorImage) {
        this.inputData = generatorImage;
    }

    async refreshAll() {
        const resultDiscriminator = await this.apiClient.getModelPrediction(this.inputData, "discriminator", this.lastLayerName);

        //change input
        this.rendererInput.changeImage(this.inputData);

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