import {changeInsideRepresentation} from "./misc.js";

class DiscriminatorController {
    constructor(calling_web_ui, apiClient, imageGridRenderer) {
        this.callingWebUI = calling_web_ui;
        this.apiClient = apiClient;
        this.imageGridRenderer = imageGridRenderer;
        this.discriminatorInputImagePixels = null;
        this.generatorOutputImage = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage("div_visualization_input_discriminator", this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    async refreshInsideDiscriminatorNew(layer_to_visualize) {
        //api call with the current layer and 'discriminator'
        const discriminator_inside_values = await this.apiClient.getModelPrediction(this.generatorOutputImage, "discriminator", layer_to_visualize);

        //change image
        changeInsideRepresentation(discriminator_inside_values, "div_visualization_inside_discriminator")
    }

    async refreshDiscriminator(newImage, resultDiscriminator) {
        //change input
        this.generatorOutputImage = newImage;
        this.imageGridRenderer.changeImage(newImage, this.discriminatorInputImagePixels);

        // print it
        let textResult = "";

        if (resultDiscriminator > 0.5) {
            textResult = "real image";
        } else {
            textResult = "fake image";
        }
        textResult += " (" + this.toPercentage(resultDiscriminator) + ")";

        document.getElementById("prediction_output_text").textContent = textResult;

        await this.refreshInsideDiscriminatorNew(document.getElementById("choice_layer_discriminator").value)
    }

    toPercentage(value) {
        return (value * 100).toFixed(2) + "%";
    }

    async updateDiscriminatorEpoch(newEpoch, shouldRefresh = true) {
        //send message to python api
        const foundEpoch = await this.apiClient.changeEpoch("discriminator", newEpoch);

        //change text
        document.getElementById("labelDiscriminatorEpochValue").textContent =
            "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUI.availableEpochs;

        if (shouldRefresh) {
            //this.refreshDiscriminator(); //todo
        }
    }
}

export default DiscriminatorController;