import {changeInsideRepresentation, toPercentage} from "./misc.js";

class DiscriminatorController {
    constructor(callingWebUI, apiClient, imageGridRenderer) {
        this.callingWebUI = callingWebUI;
        this.apiClient = apiClient;

        this.imageGridRenderer = imageGridRenderer;

        this.discriminatorInputImagePixels = null;

        this.discriminatorInputImage = null;

        this.lastLayerName = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage("div_visualization_input_discriminator", this.callingWebUI.imageSize, this.callingWebUI.imageSize);
    }

    initializeLastLayer(lastLayerName){
        this.lastLayerName = lastLayerName
    }

    async refreshInsideDiscriminator(layerToVisualize) {
        //api call with the current layer and 'discriminator'
        const discriminatorInsideValue = await this.apiClient.getModelPrediction(this.discriminatorInputImage, "discriminator", layerToVisualize);

        //change image
        changeInsideRepresentation(discriminatorInsideValue, "div_visualization_inside_discriminator")
    }

    changeInputImage(generatorImage){
        this.discriminatorInputImage = generatorImage;
    }

    async refreshDiscriminator() {
        const resultDiscriminator =  await this.apiClient.getModelPrediction(this.discriminatorInputImage , "discriminator", this.lastLayerName);

        //change input
        this.imageGridRenderer.changeImage(this.discriminatorInputImage , this.discriminatorInputImagePixels);

        //change inside
        await this.refreshInsideDiscriminator(document.getElementById("choice_layer_discriminator").value)

        // change output
        document.getElementById("prediction_output_text").textContent = this.getTextPrediction(resultDiscriminator);

    }

    getTextPrediction(resultDiscriminator){
        let textOutput = "";

        if (resultDiscriminator > 0.5) {
            textOutput = "real image";
        } else {
            textOutput = "fake image";
        }
        textOutput += " (" + toPercentage(resultDiscriminator) + ")";
        return textOutput;
    }

    async updateDiscriminatorEpoch(newEpoch, shouldRefresh = true) {
        //send message to python api
        const foundEpoch = await this.apiClient.changeEpoch("discriminator", newEpoch);

        //change text
        document.getElementById("labelDiscriminatorEpochValue").textContent = "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.callingWebUI.availableEpochs;

        if (shouldRefresh) {
            await this.refreshDiscriminator();
        }
    }
}

export default DiscriminatorController;