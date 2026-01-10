import {changeInsideRepresentation, describeMatrixShape} from "./misc.js";

class GeneratorController {
    constructor(state, apiClient, imageGridRenderer, sliderGridRenderer, discriminatorController) {
        this.state = state;
        this.apiClient = apiClient;
        this.imageGridRenderer = imageGridRenderer;
        this.sliderGridRenderer = sliderGridRenderer;
        this.discriminatorController = discriminatorController;
        this.generatorInputPixels = null;
        this.generatorImagePixels = null;
        this.slidersGrid = null;
    }

    initialize() {
        this.generatorInputPixels = this.imageGridRenderer.initializeImage(
            "grid_input_generator",
            this.state.latentSpaceSizeSqrt,
            this.state.latentSpaceSizeSqrt,
        );
        this.generatorImagePixels = this.imageGridRenderer.initializeImage(
            "grid_visual_generator",
            this.state.imageSize,
            this.state.imageSize,
        );
        this.slidersGrid = this.sliderGridRenderer.initializeGeneratorSliders(
            "sliders_grid",
            this.state.latentSpaceSizeSqrt,
            (i, j, newValue) => this.handleSliderValueChange(i, j, newValue),
        );
    }

    async refreshGeneratorAndDiscriminator() {
        const latentVectorAsMatrix = this.getInputVectorAsMatrix();

        this.imageGridRenderer.changeImage(latentVectorAsMatrix, this.generatorInputPixels);

        //const data = await this.apiClient.getResultGenerator(this.state.latentVector);
        const data_generator = await this.apiClient.getModelPrediction(this.state.latentVector, "generator", "23) conv2d");
        const data_discriminator = await this.apiClient.getModelPrediction(data_generator, "discriminator", "17) dense");

        if (!data_generator || !data_discriminator) {
            return;
        }

        //console.log('> new  inside matrix generator shape', "min max ",Math.min(...data.generated_image.flat(2)),Math.max(...data.generated_image.flat(2)))

        this.imageGridRenderer.changeImage(data_generator, this.generatorImagePixels);
        //console.log('refreshing api')

        this.refreshInsideGeneratorNew(document.getElementById("choice_layer_generator").value);

        //console.log("===> inside data", resultInside);

        await this.discriminatorController.refreshDiscriminator(data_generator, data_discriminator);
    }

    async refreshInsideGeneratorNew(layer_to_visualize) {
        //console.log("+++ refreshing generator inside", layer_to_visualize)

        //api call with the current layer and 'generator'
        this.getInputVectorAsMatrix();

        const inside_values_generator = await this.apiClient.getModelPrediction(this.state.latentVector, "generator", layer_to_visualize);

        //console.log('> new inside matrix generator shape',inside_values_generator.length, inside_values_generator[0].length, inside_values_generator[0][0].length, "min max ",Math.min(...inside_values_generator.flat(2)),Math.max(...inside_values_generator.flat(2)))
        //console.log('> new inside matrix generator',inside_values_generator)

        //change image
        changeInsideRepresentation(inside_values_generator, "grid_visual_inside_generator")
        //console.log('>> changing generator inside value')
        //describeMatrixShape(inside_values_generator)
    }

    projectTo255(x) {
        const clamped = Math.min(Math.max(x, -this.state.maxValueVisualizationInput), this.state.maxValueVisualizationInput);
        return ((clamped + this.state.maxValueVisualizationInput)
            / (2 * this.state.maxValueVisualizationInput)) * 255;
    }

    getInputVectorAsMatrix() {
        const latentVectorAsMatrix = Array.from(
            {length: this.state.latentSpaceSizeSqrt},
            () => Array(this.state.latentSpaceSizeSqrt).fill(null),
        );

        for (let i = 0; i < this.state.latentSpaceSizeSqrt; i++) {
            for (let j = 0; j < this.state.latentSpaceSizeSqrt; j++) {
                const intensity = this.state.latentVector[i * this.state.latentSpaceSizeSqrt + j];
                const intensityProjected = this.projectTo255(intensity); //between 0 black and 255 white
                latentVectorAsMatrix[i][j] = [
                    intensityProjected,
                    intensityProjected,
                    intensityProjected,
                ];
            }
        }
        return latentVectorAsMatrix;
    }

    randomizeInput() {
        const mu = parseFloat(document.getElementById("sliderMuValue").value);
        const sigma = parseFloat(document.getElementById("sliderSigmaValue").value);

        for (let i = 0; i < this.state.latentSpaceSize; i++) {
            const z = Math.sqrt(-2.0 * Math.log(Math.random()))
                * Math.cos(2.0 * Math.PI * Math.random());
            this.state.latentVector[i] = mu + sigma * z;
        }
        this.refreshSliders();
        this.refreshGeneratorAndDiscriminator();
    }

    setConstantInput() {
        const k = parseFloat(document.getElementById("sliderConstantValue").value);
        for (let i = 0; i < this.state.latentVector.length; i++) {
            this.state.latentVector[i] = k;
        }
        this.refreshSliders();
        this.refreshGeneratorAndDiscriminator();
    }

    reRandomize() {
        this.randomizeInput();
        //this.refreshGeneratorAndDiscriminator();
    }

    refreshSliders() {
        this.sliderGridRenderer.refreshSliders(
            this.slidersGrid,
            this.state.latentVector,
            this.state.latentSpaceSizeSqrt,
        );
    }

    handleSliderValueChange(i, j, newValue) {
        this.state.latentVector[i * this.state.latentSpaceSizeSqrt + j] = newValue;
        this.refreshGeneratorAndDiscriminator();
    }
}

export default GeneratorController;