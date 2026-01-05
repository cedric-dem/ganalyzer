
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

    async submitToApi() {
        const latentVectorAsMatrix = this.getInputVectorAsMatrix();
        this.imageGridRenderer.changeImage(latentVectorAsMatrix, this.generatorInputPixels);

        const data = await this.apiClient.getResultGenerator(this.state.latentVector);
        if (!data) {
            return;
        }

        this.imageGridRenderer.changeImage(data.generated_image, this.generatorImagePixels);
        this.refreshInsideGenerator();

        //todo modify
        this.state.inputValuesGenerator = {
            input: [23, 24],
            gen1: [4, 4, 45],
            gen2: [879, 7, 4, 5],
            gen3: [41, 2],
            out: [4],
        };
        this.state.inputValuesDiscriminator = {
            input: [23, 214],
            disc1: [44, 41, 145],
            disc2: [879, 7, 41, 5],
            disc3: [41, 12],
            out: [14],
        };
        const resultInside = data.inside_values;
        console.log("===> inside data", resultInside);

        this.discriminatorController.refreshDiscriminator(data.generated_image, data.result_discriminator);
    }

    refreshInsideGenerator() {
        const currentValue = document.getElementById("choice_layer_generator").value;
        //this.discriminatorController.changeInsideVisualization(true, currentValue); //todo why cant be uncommented
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
        this.submitToApi();
    }

    setConstantInput() {
        const k = parseFloat(document.getElementById("sliderConstantValue").value);
        for (let i = 0; i < this.state.latentVector.length; i++) {
            this.state.latentVector[i] = k;
        }
        this.refreshSliders();
        this.submitToApi();
    }

    reRandomize() {
        this.randomizeInput();
        this.submitToApi();
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
        this.submitToApi();
    }
}
export default GeneratorController;