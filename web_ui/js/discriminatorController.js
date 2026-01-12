import {changeInsideRepresentation, describeMatrixShape} from "./misc.js";

class DiscriminatorController {
    constructor(state, apiClient, imageGridRenderer) {
        this.state = state;
        this.apiClient = apiClient;
        this.imageGridRenderer = imageGridRenderer;
        this.discriminatorInputImagePixels = null;
        this.generatorOutputImage = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage(
            "grid_input_discriminator",
            this.state.imageSize,
            this.state.imageSize,
        );
    }

    async refreshInsideDiscriminatorNew(layer_to_visualize) {
        //console.log("+++ refreshing discriminator inside", layer_to_visualize)

        //api call with the current layer and 'discriminator'
        const generated_image = this.generatorOutputImage;

        const discriminator_inside_values = await this.apiClient.getModelPrediction(generated_image, "discriminator", layer_to_visualize);

        //console.log('> new inside matrix discriminator shape',discriminator_inside_values.length, discriminator_inside_values[0].length, discriminator_inside_values[0][0].length);
        //console.log('> new inside matrix discriminator shape', discriminator_inside_values);

        //change image
        changeInsideRepresentation(discriminator_inside_values, "grid_visual_inside_discriminator")
        //console.log('>> changing discriminator inside value')
        //describeMatrixShape(discriminator_inside_values)

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

        //this.refreshInsideDiscriminator();
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
            "Epoch : " + newEpoch + "(" + foundEpoch + ")" + "/" + this.state.availableEpochs;
        //get_result_discriminator()

        if (shouldRefresh) {
            //this.refreshDiscriminator();
        }
    }
}

export default DiscriminatorController;