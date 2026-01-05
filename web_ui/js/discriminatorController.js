
class DiscriminatorController {
    constructor(state, imageGridRenderer) {
        this.state = state;
        this.imageGridRenderer = imageGridRenderer;
        this.discriminatorInputImagePixels = null;
    }

    initialize() {
        this.discriminatorInputImagePixels = this.imageGridRenderer.initializeImage(
            "grid_input_discriminator",
            this.state.imageSize,
            this.state.imageSize,
        );
    }

    changeInsideVisualization(isGenerator, selectedValue) {
        console.log("change inside visualization ", isGenerator, selectedValue);

        let valuesToDisplay;
        let location;

        if (isGenerator) {
            valuesToDisplay = this.state.inputValuesGenerator?.[selectedValue];
            location = "grid_visual_inside_discriminator";
        } else {
            valuesToDisplay = this.state.inputValuesDiscriminator?.[selectedValue];
            location = "grid_visual_inside_generator";
        }

        console.log(" >> values to add in input field",
            valuesToDisplay,
            "location : ",
            location,
        );

        // TODO empty location
        // TODO initialize div
        // TODO fill with all_values_list[selected_value]
    }

    refreshInsideDiscriminator() {
        const currentValue = document.getElementById("choice_layer_discriminator").value;
        this.changeInsideVisualization(false, currentValue);
    }

    refreshDiscriminator(newImage, resultDiscriminator) {
        //change input
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

        this.refreshInsideDiscriminator();
    }

    toPercentage(value) {
        return (value * 100).toFixed(2) + "%";
    }
}
export default DiscriminatorController;